//! C FFI bindings for yardstick-rs
//!
//! Exposes the query rewriter to C/C++ consumers like the DuckDB extension.
//!
//! Safety: These functions are `extern "C"` and expect valid C strings.
//! Callers must ensure pointers are valid. Documented in header.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::ffi::{CStr, CString};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::os::raw::c_char;
use std::path::{Path, PathBuf};
use std::ptr;
use std::sync::Mutex;

use once_cell::sync::Lazy;

use crate::config::{load_from_directory, load_from_file, load_from_string, parse_sql_model};
use crate::core::SemanticGraph;
use crate::sql::{
    expand_aggregate_with_at, expand_curly_braces, get_measure_aggregation, has_aggregate_function,
    has_as_measure, has_at_syntax, has_curly_brace_measure, process_create_view, QueryRewriter,
};

/// Global semantic graph state (thread-safe)
static SEMANTIC_GRAPH: Lazy<Mutex<SemanticGraph>> = Lazy::new(|| Mutex::new(SemanticGraph::new()));

/// Active model for METRIC/DIMENSION/SEGMENT additions (set by CREATE MODEL or USE)
static ACTIVE_MODEL: Lazy<Mutex<Option<String>>> = Lazy::new(|| Mutex::new(None));

/// Result from rewrite operation
#[repr(C)]
pub struct YardstickRewriteResult {
    /// Rewritten SQL (null if error)
    pub sql: *mut c_char,
    /// Error message (null if success)
    pub error: *mut c_char,
    /// Whether the query was rewritten (false = passthrough)
    pub was_rewritten: bool,
}

/// Load semantic models from YAML string
///
/// Returns null on success, error message on failure.
/// Caller must free the returned string with `yardstick_free`.
#[no_mangle]
pub extern "C" fn yardstick_load_yaml(yaml: *const c_char) -> *mut c_char {
    if yaml.is_null() {
        return to_c_string("Error: null yaml pointer");
    }

    let yaml_str = unsafe {
        match CStr::from_ptr(yaml).to_str() {
            Ok(s) => s,
            Err(e) => return to_c_string(&format!("Error: invalid UTF-8: {e}")),
        }
    };

    match load_from_string(yaml_str) {
        Ok(new_graph) => {
            let mut graph = SEMANTIC_GRAPH.lock().unwrap();
            // Merge new models into existing graph
            for model in new_graph.models() {
                if let Err(e) = graph.add_model(model.clone()) {
                    return to_c_string(&format!("Error adding model: {e}"));
                }
            }
            ptr::null_mut() // Success
        }
        Err(e) => to_c_string(&format!("Error: {e}")),
    }
}

/// Load semantic models from a file or directory path
///
/// Returns null on success, error message on failure.
/// Caller must free the returned string with `yardstick_free`.
#[no_mangle]
pub extern "C" fn yardstick_load_file(path: *const c_char) -> *mut c_char {
    if path.is_null() {
        return to_c_string("Error: null path pointer");
    }

    let path_str = unsafe {
        match CStr::from_ptr(path).to_str() {
            Ok(s) => s,
            Err(e) => return to_c_string(&format!("Error: invalid UTF-8: {e}")),
        }
    };

    let path = Path::new(path_str);

    // Check if path exists
    if !path.exists() {
        return to_c_string(&format!("Error: path does not exist: {path_str}"));
    }

    let result = if path.is_dir() {
        load_from_directory(path)
    } else {
        load_from_file(path)
    };

    match result {
        Ok(new_graph) => {
            let mut graph = SEMANTIC_GRAPH.lock().unwrap();
            // Merge new models into existing graph
            for model in new_graph.models() {
                if let Err(e) = graph.add_model(model.clone()) {
                    return to_c_string(&format!("Error adding model: {e}"));
                }
            }
            ptr::null_mut() // Success
        }
        Err(e) => to_c_string(&format!("Error: {e}")),
    }
}

/// Clear all loaded semantic models
#[no_mangle]
pub extern "C" fn yardstick_clear() {
    let mut graph = SEMANTIC_GRAPH.lock().unwrap();
    *graph = SemanticGraph::new();
}

/// Define a semantic model from SQL definition format
///
/// Parses the definition, saves to file, and loads into current session.
/// If `replace` is true, removes any existing model with the same name from the file.
///
/// Returns null on success, error message on failure.
/// Caller must free the returned string with `yardstick_free`.
#[no_mangle]
pub extern "C" fn yardstick_define(
    definition_sql: *const c_char,
    db_path: *const c_char,
    replace: bool,
) -> *mut c_char {
    if definition_sql.is_null() {
        return to_c_string("Error: null definition_sql pointer");
    }

    let sql_str = unsafe {
        match CStr::from_ptr(definition_sql).to_str() {
            Ok(s) => s,
            Err(e) => return to_c_string(&format!("Error: invalid UTF-8: {e}")),
        }
    };

    // Parse the definition to validate and get model name
    let model = match parse_sql_model(sql_str) {
        Ok(m) => m,
        Err(e) => return to_c_string(&format!("Error parsing definition: {e}")),
    };

    let model_name = model.name.clone();

    // Determine the definitions file path
    let definitions_path = get_definitions_path(db_path);

    // Handle OR REPLACE: read existing file, remove model if exists
    if replace {
        if let Err(e) = remove_model_from_file(&definitions_path, &model_name) {
            return to_c_string(&format!("Error removing existing model: {e}"));
        }
    }

    // Append definition to file
    if let Err(e) = append_definition_to_file(&definitions_path, sql_str) {
        return to_c_string(&format!("Error writing to definitions file: {e}"));
    }

    // Load model into current session
    let mut graph = SEMANTIC_GRAPH.lock().unwrap();
    if let Err(e) = graph.add_model(model) {
        return to_c_string(&format!("Error adding model to session: {e}"));
    }

    // Set this model as the active model for subsequent METRIC/DIMENSION additions
    *ACTIVE_MODEL.lock().unwrap() = Some(model_name);

    ptr::null_mut() // Success
}

/// Get the definitions file path based on database path
fn get_definitions_path(db_path: *const c_char) -> PathBuf {
    if db_path.is_null() {
        // In-memory database: use current directory
        return PathBuf::from("./yardstick_definitions.sql");
    }

    let path_str = unsafe {
        match CStr::from_ptr(db_path).to_str() {
            Ok(s) => s,
            Err(_) => return PathBuf::from("./yardstick_definitions.sql"),
        }
    };

    if path_str.is_empty() || path_str == ":memory:" {
        return PathBuf::from("./yardstick_definitions.sql");
    }

    // Replace .duckdb extension with .yardstick.sql
    let db_path = Path::new(path_str);
    let stem = db_path.file_stem().unwrap_or_default();
    let parent = db_path.parent().unwrap_or(Path::new("."));
    parent.join(format!("{}.yardstick.sql", stem.to_string_lossy()))
}

/// Remove a model definition from the file by name
fn remove_model_from_file(path: &Path, model_name: &str) -> std::io::Result<()> {
    if !path.exists() {
        return Ok(()); // Nothing to remove
    }

    let content = fs::read_to_string(path)?;
    let mut result = String::new();
    let mut skip_until_next_model = false;
    let model_pattern = "MODEL".to_string();
    let name_pattern = format!("name {model_name}");
    let name_pattern_comma = format!("name {model_name},");

    for line in content.lines() {
        let line_trimmed = line.trim().to_uppercase();

        // Check if this is a MODEL statement
        if line_trimmed.starts_with(&model_pattern) {
            // Check if this model has the name we're looking for
            let line_lower = line.to_lowercase();
            if line_lower.contains(&name_pattern.to_lowercase())
                || line_lower.contains(&name_pattern_comma.to_lowercase())
            {
                skip_until_next_model = true;
                continue;
            }
            skip_until_next_model = false;
        }

        // If we encounter another statement type, stop skipping
        if skip_until_next_model
            && (line_trimmed.starts_with("MODEL")
                || line_trimmed.starts_with("--")
                || line_trimmed.is_empty())
        {
            if line_trimmed.starts_with("MODEL")
                && !line.to_lowercase().contains(&name_pattern.to_lowercase())
            {
                skip_until_next_model = false;
            } else if line_trimmed.is_empty() || line_trimmed.starts_with("--") {
                // Skip empty lines and comments between removed statements
                continue;
            }
        }

        if !skip_until_next_model {
            result.push_str(line);
            result.push('\n');
        }
    }

    fs::write(path, result.trim_end())?;
    Ok(())
}

/// Append a definition to the file
fn append_definition_to_file(path: &Path, definition: &str) -> std::io::Result<()> {
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;

    // Add newlines for separation if file is not empty
    if path.exists() && fs::metadata(path)?.len() > 0 {
        writeln!(file)?;
        writeln!(file)?;
    }

    writeln!(file, "{}", definition.trim())?;
    Ok(())
}

/// Load definitions from file if it exists (for auto-load on extension start)
///
/// Returns null on success (including when file doesn't exist), error message on failure.
/// Caller must free the returned string with `yardstick_free`.
#[no_mangle]
pub extern "C" fn yardstick_autoload(db_path: *const c_char) -> *mut c_char {
    let definitions_path = get_definitions_path(db_path);

    if !definitions_path.exists() {
        return ptr::null_mut(); // No file to load, success
    }

    // Read and parse the definitions file
    let content = match fs::read_to_string(&definitions_path) {
        Ok(c) => c,
        Err(e) => return to_c_string(&format!("Error reading definitions file: {e}")),
    };

    if content.trim().is_empty() {
        return ptr::null_mut(); // Empty file, success
    }

    // Parse each model definition in the file
    // Split on MODEL keyword to handle multiple definitions
    let mut graph = SEMANTIC_GRAPH.lock().unwrap();

    for block in split_definitions(&content) {
        if block.trim().is_empty() {
            continue;
        }
        match parse_sql_model(block) {
            Ok(model) => {
                if let Err(e) = graph.add_model(model) {
                    return to_c_string(&format!("Error loading model: {e}"));
                }
            }
            Err(e) => {
                // Log but don't fail on parse errors for individual models
                eprintln!("Warning: failed to parse model definition: {e}");
            }
        }
    }

    ptr::null_mut() // Success
}

/// Split content into individual model definitions
fn split_definitions(content: &str) -> Vec<&str> {
    let mut definitions = Vec::new();
    let mut start = 0;

    // Find each MODEL keyword and split there
    let content_upper = content.to_uppercase();
    let mut search_start = 0;

    while let Some(pos) = content_upper[search_start..].find("MODEL") {
        let actual_pos = search_start + pos;

        // Check this is actually the start of a MODEL statement (not inside a word)
        let is_start =
            actual_pos == 0 || !content.as_bytes()[actual_pos - 1].is_ascii_alphanumeric();
        let is_followed_by_space = actual_pos + 5 < content.len()
            && (content.as_bytes()[actual_pos + 5] == b' '
                || content.as_bytes()[actual_pos + 5] == b'('
                || content.as_bytes()[actual_pos + 5] == b'\t'
                || content.as_bytes()[actual_pos + 5] == b'\n');

        if is_start && is_followed_by_space {
            if start < actual_pos && start > 0 {
                definitions.push(&content[start..actual_pos]);
            }
            start = actual_pos;
        }

        search_start = actual_pos + 1;
    }

    // Don't forget the last definition
    if start < content.len() {
        definitions.push(&content[start..]);
    }

    definitions
}

/// Add a metric/dimension/segment to the most recently created model
///
/// definition_sql: The definition in nom format (e.g., "METRIC (name revenue, agg sum, sql amount)")
/// db_path: Path to database file for persistence (null for in-memory)
/// is_replace: If true, replace existing metric/dimension/segment with the same name
///
/// Supports syntaxes:
/// - `METRIC (name foo, ...)` - adds to active model
/// - `METRIC model.foo (...)` - adds to specified model
/// - `METRIC foo AS SUM(x)` - adds to active model
/// - `METRIC model.foo AS SUM(x)` - adds to specified model
///
/// Returns null on success, error message on failure.
#[no_mangle]
pub extern "C" fn yardstick_add_definition(
    definition_sql: *const c_char,
    db_path: *const c_char,
    is_replace: bool,
) -> *mut c_char {
    use crate::config::parse_sql_model;

    if definition_sql.is_null() {
        return to_c_string("Error: null definition_sql pointer");
    }

    let sql_str = unsafe {
        match CStr::from_ptr(definition_sql).to_str() {
            Ok(s) => s,
            Err(e) => return to_c_string(&format!("Error: invalid UTF-8: {e}")),
        }
    };

    // Parse to determine what type it is and extract properties
    let sql_trimmed = sql_str.trim();
    let sql_upper = sql_trimmed.to_uppercase();

    let mut graph = SEMANTIC_GRAPH.lock().unwrap();

    // Check for model.name syntax: "METRIC model.name (...)" or "DIMENSION model.name (...)"
    // Extract model name if present, otherwise use ACTIVE_MODEL
    let (target_model_name, adjusted_sql) = extract_model_prefix(sql_trimmed);

    let model_name = if let Some(explicit_model) = target_model_name {
        // Verify the model exists
        if graph.get_model(&explicit_model).is_none() {
            return to_c_string(&format!("Error: model '{explicit_model}' not found"));
        }
        explicit_model
    } else {
        // Use ACTIVE_MODEL or fall back to last model
        let active = ACTIVE_MODEL.lock().unwrap();
        if let Some(ref name) = *active {
            name.clone()
        } else {
            // Fall back to last model
            let model_names: Vec<String> = graph.models().map(|m| m.name.clone()).collect();
            if model_names.is_empty() {
                return to_c_string("Error: no model defined yet. Create a model first with SEMANTIC CREATE MODEL, or use SEMANTIC USE <model>.");
            }
            model_names.last().unwrap().clone()
        }
    };

    // Get the model to modify
    let model = match graph.get_model(&model_name) {
        Some(m) => m.clone(),
        None => return to_c_string(&format!("Error: could not find model '{model_name}'")),
    };

    // Parse the definition using a dummy model wrapper
    let dummy_sql = format!("MODEL (name {model_name}, table dummy);\n{adjusted_sql}");
    let parsed = match parse_sql_model(&dummy_sql) {
        Ok(m) => m,
        Err(e) => return to_c_string(&format!("Error parsing definition: {e}")),
    };

    // Extract what was added and update the model
    let mut updated_model = model.clone();

    if sql_upper.starts_with("METRIC") {
        for metric in parsed.metrics {
            if is_replace {
                // Remove existing metric with same name
                updated_model.metrics.retain(|m| m.name != metric.name);
            }
            updated_model.metrics.push(metric);
        }
    } else if sql_upper.starts_with("DIMENSION") {
        for dim in parsed.dimensions {
            if is_replace {
                // Remove existing dimension with same name
                updated_model.dimensions.retain(|d| d.name != dim.name);
            }
            updated_model.dimensions.push(dim);
        }
    } else if sql_upper.starts_with("SEGMENT") {
        for seg in parsed.segments {
            if is_replace {
                // Remove existing segment with same name
                updated_model.segments.retain(|s| s.name != seg.name);
            }
            updated_model.segments.push(seg);
        }
    }

    // add_model will overwrite since it uses HashMap::insert
    if let Err(e) = graph.add_model(updated_model) {
        return to_c_string(&format!("Error updating model: {e}"));
    }

    // Append to definitions file
    let definitions_path = get_definitions_path(db_path);
    if let Err(e) = append_definition_to_file(&definitions_path, sql_str) {
        return to_c_string(&format!("Error writing to definitions file: {e}"));
    }

    ptr::null_mut() // Success
}

/// Extract model prefix from "METRIC model.name (...)" or "METRIC model.name AS expr" syntax
/// Returns (Some(model), adjusted_sql) if prefix found, (None, original_sql) otherwise
fn extract_model_prefix(sql: &str) -> (Option<String>, String) {
    let sql_upper = sql.to_uppercase();

    // Find the keyword (METRIC, DIMENSION, SEGMENT)
    let keyword = if sql_upper.starts_with("METRIC") {
        "METRIC"
    } else if sql_upper.starts_with("DIMENSION") {
        "DIMENSION"
    } else if sql_upper.starts_with("SEGMENT") {
        "SEGMENT"
    } else {
        return (None, sql.to_string());
    };

    // Get everything after the keyword
    let rest = sql[keyword.len()..].trim_start();

    // Check for model.name pattern - could be followed by ( or AS
    // Look for a dot in the first identifier
    let first_space_or_paren = rest
        .find(|c: char| c.is_whitespace() || c == '(')
        .unwrap_or(rest.len());
    let first_token = &rest[..first_space_or_paren];

    if let Some(dot_pos) = first_token.find('.') {
        let model_name = first_token[..dot_pos].trim();
        let field_name = first_token[dot_pos + 1..].trim();
        let after_token = rest[first_space_or_paren..].trim_start();

        // Check what follows: ( for paren syntax, or AS for simple syntax
        if after_token.starts_with('(') {
            // Paren syntax: "model.name (props...)"
            let paren_content = after_token;
            let adjusted = if let Some(stripped) = paren_content.strip_prefix('(') {
                let inner = stripped.trim_start();
                format!("{keyword} (name {field_name}, {inner}")
            } else {
                format!("{keyword} {rest}")
            };
            return (Some(model_name.to_string()), adjusted);
        } else if after_token.to_uppercase().starts_with("AS ") {
            // AS syntax: "model.name AS expr"
            let after_as = &after_token[2..].trim_start();
            let adjusted = format!("{keyword} {field_name} AS {after_as}");
            return (Some(model_name.to_string()), adjusted);
        }
    }

    (None, sql.to_string())
}

/// Set the active model for subsequent METRIC/DIMENSION/SEGMENT additions
///
/// Returns null on success, error message on failure.
#[no_mangle]
pub extern "C" fn yardstick_use(model_name: *const c_char) -> *mut c_char {
    if model_name.is_null() {
        return to_c_string("Error: null model_name pointer");
    }

    let name_str = unsafe {
        match CStr::from_ptr(model_name).to_str() {
            Ok(s) => s,
            Err(e) => return to_c_string(&format!("Error: invalid UTF-8: {e}")),
        }
    };

    let name = name_str.trim();
    if name.is_empty() {
        return to_c_string("Error: model name cannot be empty");
    }

    // Verify the model exists
    let graph = SEMANTIC_GRAPH.lock().unwrap();
    if graph.get_model(name).is_none() {
        let available: Vec<&str> = graph.models().map(|m| m.name.as_str()).collect();
        return to_c_string(&format!(
            "Error: model '{}' not found. Available models: {}",
            name,
            if available.is_empty() {
                "(none)".to_string()
            } else {
                available.join(", ")
            }
        ));
    }
    drop(graph); // Release lock before acquiring ACTIVE_MODEL lock

    // Set active model
    *ACTIVE_MODEL.lock().unwrap() = Some(name.to_string());

    ptr::null_mut() // Success
}

/// Check if a table name is a registered semantic model
#[no_mangle]
pub extern "C" fn yardstick_is_model(table_name: *const c_char) -> bool {
    if table_name.is_null() {
        return false;
    }

    let name = unsafe {
        match CStr::from_ptr(table_name).to_str() {
            Ok(s) => s,
            Err(_) => return false,
        }
    };

    let graph = SEMANTIC_GRAPH.lock().unwrap();
    graph.get_model(name).is_some()
}

/// Get list of registered model names (comma-separated)
///
/// Caller must free the returned string with `yardstick_free`.
#[no_mangle]
pub extern "C" fn yardstick_list_models() -> *mut c_char {
    let graph = SEMANTIC_GRAPH.lock().unwrap();
    let names: Vec<&str> = graph.models().map(|m| m.name.as_str()).collect();
    to_c_string(&names.join(","))
}

/// Rewrite a SQL query using semantic definitions
///
/// Returns a YardstickRewriteResult struct. Caller must free with `yardstick_free_result`.
#[no_mangle]
pub extern "C" fn yardstick_rewrite(sql: *const c_char) -> YardstickRewriteResult {
    if sql.is_null() {
        return YardstickRewriteResult {
            sql: ptr::null_mut(),
            error: to_c_string("Error: null sql pointer"),
            was_rewritten: false,
        };
    }

    let sql_str = unsafe {
        match CStr::from_ptr(sql).to_str() {
            Ok(s) => s,
            Err(e) => {
                return YardstickRewriteResult {
                    sql: ptr::null_mut(),
                    error: to_c_string(&format!("Error: invalid UTF-8: {e}")),
                    was_rewritten: false,
                }
            }
        }
    };

    let graph = SEMANTIC_GRAPH.lock().unwrap();

    // Check if query references any semantic models
    if !query_references_models(sql_str, &graph) {
        // Passthrough - not a semantic query
        return YardstickRewriteResult {
            sql: to_c_string(sql_str),
            error: ptr::null_mut(),
            was_rewritten: false,
        };
    }

    // Rewrite the query
    let rewriter = QueryRewriter::new(&graph);
    match rewriter.rewrite(sql_str) {
        Ok(rewritten) => YardstickRewriteResult {
            sql: to_c_string(&rewritten),
            error: ptr::null_mut(),
            was_rewritten: true,
        },
        Err(e) => YardstickRewriteResult {
            sql: ptr::null_mut(),
            error: to_c_string(&format!("Error: {e}")),
            was_rewritten: false,
        },
    }
}

/// Free a string returned by yardstick functions
#[no_mangle]
pub extern "C" fn yardstick_free(ptr: *mut c_char) {
    if !ptr.is_null() {
        unsafe {
            drop(CString::from_raw(ptr));
        }
    }
}

/// Free a YardstickRewriteResult
#[no_mangle]
pub extern "C" fn yardstick_free_result(result: YardstickRewriteResult) {
    yardstick_free(result.sql);
    yardstick_free(result.error);
}

// Helper: convert Rust string to C string
fn to_c_string(s: &str) -> *mut c_char {
    match CString::new(s) {
        Ok(cs) => cs.into_raw(),
        Err(_) => ptr::null_mut(),
    }
}

// Helper: check if SQL references any registered models
fn query_references_models(sql: &str, graph: &SemanticGraph) -> bool {
    let sql_lower = sql.to_lowercase();

    for model in graph.models() {
        let model_lower = model.name.to_lowercase();

        // Check for FROM model or JOIN model patterns
        if sql_lower.contains(&format!("from {model_lower}"))
            || sql_lower.contains(&format!("from {model_lower} "))
            || sql_lower.contains(&format!("join {model_lower}"))
            || sql_lower.contains(&format!("join {model_lower} "))
            // Also check for model.column references
            || sql_lower.contains(&format!("{model_lower}."))
        {
            return true;
        }
    }

    false
}

// =============================================================================
// Julian Hyde "Measures in SQL" FFI Functions
// =============================================================================

/// Result from processing CREATE VIEW with AS MEASURE
#[repr(C)]
pub struct YardstickCreateViewResult {
    /// Whether this had AS MEASURE columns
    pub is_measure_view: bool,
    /// View name (null if not a view)
    pub view_name: *mut c_char,
    /// Clean SQL with AS MEASURE removed (to execute)
    pub clean_sql: *mut c_char,
    /// Error message (null if success)
    pub error: *mut c_char,
}

/// Result from expanding AGGREGATE() calls
#[repr(C)]
pub struct YardstickAggregateResult {
    /// Whether any AGGREGATE() calls were expanded
    pub had_aggregate: bool,
    /// Expanded SQL
    pub expanded_sql: *mut c_char,
    /// Error message (null if success)
    pub error: *mut c_char,
}

/// Check if SQL contains "AS MEASURE" pattern
///
/// Quick check before calling process_create_view
#[no_mangle]
pub extern "C" fn yardstick_has_as_measure(sql: *const c_char) -> bool {
    if sql.is_null() {
        return false;
    }

    let sql_str = unsafe {
        match CStr::from_ptr(sql).to_str() {
            Ok(s) => s,
            Err(_) => return false,
        }
    };

    has_as_measure(sql_str)
}

/// Check if SQL contains AGGREGATE() function
///
/// Quick check before calling expand_aggregate
#[no_mangle]
pub extern "C" fn yardstick_has_aggregate(sql: *const c_char) -> bool {
    if sql.is_null() {
        return false;
    }

    let sql_str = unsafe {
        match CStr::from_ptr(sql).to_str() {
            Ok(s) => s,
            Err(_) => return false,
        }
    };

    has_aggregate_function(sql_str)
}

/// Check if SQL contains curly brace measure syntax: `{column}`
///
/// Quick check before calling expand_curly_braces
#[no_mangle]
pub extern "C" fn yardstick_has_curly_brace(sql: *const c_char) -> bool {
    if sql.is_null() {
        return false;
    }

    let sql_str = unsafe {
        match CStr::from_ptr(sql).to_str() {
            Ok(s) => s,
            Err(_) => return false,
        }
    };

    has_curly_brace_measure(sql_str)
}

/// Expand curly brace syntax `{column}` to `AGGREGATE(column)`
///
/// Returns the expanded SQL string. Caller must free with yardstick_free.
#[no_mangle]
pub extern "C" fn yardstick_expand_curly_braces(sql: *const c_char) -> *mut c_char {
    if sql.is_null() {
        return ptr::null_mut();
    }

    let sql_str = unsafe {
        match CStr::from_ptr(sql).to_str() {
            Ok(s) => s,
            Err(_) => return ptr::null_mut(),
        }
    };

    let expanded = expand_curly_braces(sql_str);
    to_c_string(&expanded)
}

/// Process CREATE VIEW statement, extracting AS MEASURE definitions
///
/// Input: `CREATE VIEW foo AS SELECT a, SUM(b) AS MEASURE revenue FROM t`
/// Output: Clean SQL (AS MEASURE removed) and measures stored in catalog
///
/// Caller must free with yardstick_free_create_view_result
#[no_mangle]
pub extern "C" fn yardstick_process_create_view(sql: *const c_char) -> YardstickCreateViewResult {
    if sql.is_null() {
        return YardstickCreateViewResult {
            is_measure_view: false,
            view_name: ptr::null_mut(),
            clean_sql: ptr::null_mut(),
            error: to_c_string("Error: null sql pointer"),
        };
    }

    let sql_str = unsafe {
        match CStr::from_ptr(sql).to_str() {
            Ok(s) => s,
            Err(e) => {
                return YardstickCreateViewResult {
                    is_measure_view: false,
                    view_name: ptr::null_mut(),
                    clean_sql: ptr::null_mut(),
                    error: to_c_string(&format!("Error: invalid UTF-8: {e}")),
                };
            }
        }
    };

    let result = process_create_view(sql_str);

    YardstickCreateViewResult {
        is_measure_view: result.is_measure_view,
        view_name: result.view_name.map(|s| to_c_string(&s)).unwrap_or(ptr::null_mut()),
        clean_sql: to_c_string(&result.clean_sql),
        error: result.error.map(|s| to_c_string(&s)).unwrap_or(ptr::null_mut()),
    }
}

/// Expand AGGREGATE() function calls in a SELECT statement
///
/// Input: `SELECT status, AGGREGATE(revenue) FROM orders_summary GROUP BY status`
/// Output: Expanded SQL with actual aggregations
///
/// Also handles AT syntax:
/// - `AGGREGATE(revenue) AT (ALL status)` -> scalar subquery
/// - `AGGREGATE(revenue) AT (SET dim = expr)` -> correlated subquery
/// - `AGGREGATE(revenue) AT (WHERE cond)` -> filtered subquery
///
/// Caller must free with yardstick_free_aggregate_result
#[no_mangle]
pub extern "C" fn yardstick_expand_aggregate(sql: *const c_char) -> YardstickAggregateResult {
    if sql.is_null() {
        return YardstickAggregateResult {
            had_aggregate: false,
            expanded_sql: ptr::null_mut(),
            error: to_c_string("Error: null sql pointer"),
        };
    }

    let sql_str = unsafe {
        match CStr::from_ptr(sql).to_str() {
            Ok(s) => s,
            Err(e) => {
                return YardstickAggregateResult {
                    had_aggregate: false,
                    expanded_sql: ptr::null_mut(),
                    error: to_c_string(&format!("Error: invalid UTF-8: {e}")),
                };
            }
        }
    };

    // Use expand_aggregate_with_at which handles both plain AGGREGATE and AGGREGATE AT
    let result = expand_aggregate_with_at(sql_str);

    YardstickAggregateResult {
        had_aggregate: result.had_aggregate,
        expanded_sql: to_c_string(&result.expanded_sql),
        error: result.error.map(|s| to_c_string(&s)).unwrap_or(ptr::null_mut()),
    }
}

/// Check if SQL contains AT syntax (AGGREGATE() AT ())
#[no_mangle]
pub extern "C" fn yardstick_has_at_syntax(sql: *const c_char) -> bool {
    if sql.is_null() {
        return false;
    }

    let sql_str = unsafe {
        match CStr::from_ptr(sql).to_str() {
            Ok(s) => s,
            Err(_) => return false,
        }
    };

    has_at_syntax(sql_str)
}

/// Result from looking up measure aggregation
#[repr(C)]
pub struct YardstickMeasureAggResult {
    /// Aggregation function name (lowercase: "sum", "count", etc.) or null if not found
    pub aggregation: *mut c_char,
    /// View name where the measure was found, or null if not found
    pub view_name: *mut c_char,
}

/// Look up the aggregation function for a measure column
///
/// Given a column name (e.g., "revenue"), searches all measure views for
/// a matching measure and returns its aggregation function (e.g., "sum").
///
/// Caller must free with yardstick_free_measure_agg_result
#[no_mangle]
pub extern "C" fn yardstick_get_measure_aggregation(
    column_name: *const c_char,
) -> YardstickMeasureAggResult {
    if column_name.is_null() {
        return YardstickMeasureAggResult {
            aggregation: ptr::null_mut(),
            view_name: ptr::null_mut(),
        };
    }

    let name_str = unsafe {
        match CStr::from_ptr(column_name).to_str() {
            Ok(s) => s,
            Err(_) => {
                return YardstickMeasureAggResult {
                    aggregation: ptr::null_mut(),
                    view_name: ptr::null_mut(),
                };
            }
        }
    };

    match get_measure_aggregation(name_str) {
        Some((agg_fn, view_name)) => YardstickMeasureAggResult {
            aggregation: to_c_string(&agg_fn),
            view_name: to_c_string(&view_name),
        },
        None => YardstickMeasureAggResult {
            aggregation: ptr::null_mut(),
            view_name: ptr::null_mut(),
        },
    }
}

/// Free a YardstickMeasureAggResult
#[no_mangle]
pub extern "C" fn yardstick_free_measure_agg_result(result: YardstickMeasureAggResult) {
    yardstick_free(result.aggregation);
    yardstick_free(result.view_name);
}

/// Free a YardstickCreateViewResult
#[no_mangle]
pub extern "C" fn yardstick_free_create_view_result(result: YardstickCreateViewResult) {
    yardstick_free(result.view_name);
    yardstick_free(result.clean_sql);
    yardstick_free(result.error);
}

/// Free a YardstickAggregateResult
#[no_mangle]
pub extern "C" fn yardstick_free_aggregate_result(result: YardstickAggregateResult) {
    yardstick_free(result.expanded_sql);
    yardstick_free(result.error);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;

    #[test]
    fn test_load_and_rewrite() {
        // Clear any existing state
        yardstick_clear();

        // Load a model
        let yaml = CString::new(
            r#"
models:
  - name: orders
    table: orders
    primary_key: order_id
    dimensions:
      - name: status
        type: categorical
    metrics:
      - name: revenue
        agg: sum
        sql: amount
"#,
        )
        .unwrap();

        let result = yardstick_load_yaml(yaml.as_ptr());
        assert!(result.is_null()); // Success

        // Check model is registered
        let name = CString::new("orders").unwrap();
        assert!(yardstick_is_model(name.as_ptr()));

        // Rewrite a query
        let sql = CString::new("SELECT orders.revenue, orders.status FROM orders").unwrap();
        let result = yardstick_rewrite(sql.as_ptr());

        assert!(result.error.is_null());
        assert!(result.was_rewritten);

        let rewritten = unsafe { CStr::from_ptr(result.sql).to_str().unwrap() };
        assert!(rewritten.contains("SUM"));

        yardstick_free_result(result);
    }

    #[test]
    fn test_passthrough() {
        yardstick_clear();

        // Query without semantic models should pass through
        let sql = CString::new("SELECT * FROM some_table").unwrap();
        let result = yardstick_rewrite(sql.as_ptr());

        assert!(result.error.is_null());
        assert!(!result.was_rewritten);

        yardstick_free_result(result);
    }
}
