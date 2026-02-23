//! Julian Hyde's "Measures in SQL" support
//!
//! Implements:
//! - `AS MEASURE` in CREATE VIEW (arXiv:2406.00251)
//! - `AGGREGATE()` function expansion
//! - `AT (ALL dim)`, `AT (SET dim = expr)` context modifiers
//!
//! Reference: https://arxiv.org/abs/2406.00251

use std::collections::{HashMap, HashSet};
use std::sync::Mutex;

use nom::{
    branch::alt,
    bytes::complete::{tag_no_case, take_while, take_while1},
    character::complete::{char, multispace0, multispace1},
    combinator::{opt, recognize},
    sequence::{delimited, pair, tuple},
    IResult,
};
use once_cell::sync::Lazy;

use crate::error::{Result, YardstickError};
use crate::parser_ffi::{self, SelectInfo};

// =============================================================================
// Data Structures
// =============================================================================

/// Stored measure definition from CREATE VIEW ... AS MEASURE
#[derive(Debug, Clone)]
pub struct ViewMeasure {
    pub column_name: String,
    pub expression: String,
    /// Whether this measure can be re-aggregated (false for COUNT DISTINCT)
    pub is_decomposable: bool,
}

/// Stored view with measures
#[derive(Debug, Clone)]
pub struct MeasureView {
    pub view_name: String,
    pub measures: Vec<ViewMeasure>,
    pub base_query: String,
    /// Base table name for deferred evaluation of non-decomposable measures
    pub base_table: Option<String>,
    /// Base relation SQL used to recompute non-decomposable measures
    pub base_relation_sql: Option<String>,
    /// Map of dimension aliases to their expressions (from view SELECT)
    pub dimension_exprs: HashMap<String, String>,
    /// GROUP BY columns from the view definition (if present)
    pub group_by_cols: Vec<String>,
}

/// Global storage for measure views (in-memory catalog)
static MEASURE_VIEWS: Lazy<Mutex<HashMap<String, MeasureView>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

const DEFAULT_CONTEXT_MARKER: &str = "/*YARDSTICK_DEFAULT*/";

/// Result from processing CREATE VIEW with AS MEASURE
#[derive(Debug)]
pub struct CreateViewResult {
    pub is_measure_view: bool,
    pub view_name: Option<String>,
    pub clean_sql: String,
    pub measures: Vec<ViewMeasure>,
    pub error: Option<String>,
}

/// Result from expanding AGGREGATE() calls
#[derive(Debug)]
pub struct AggregateExpandResult {
    pub had_aggregate: bool,
    pub expanded_sql: String,
    pub error: Option<String>,
}

/// Context modifier for AGGREGATE() AT (...) syntax
#[derive(Debug, Clone, PartialEq)]
pub enum ContextModifier {
    /// AT (ALL) - grand total over entire table, no filters
    AllGlobal,
    /// AT (ALL dimension) - remove dimension from context
    All(String),
    /// AT (SET dimension = expr) - change dimension value
    Set(String, String),
    /// AT (WHERE condition) - filter to condition
    Where(String),
    /// AT (VISIBLE) - respect outer query's WHERE clause
    Visible,
}

/// Information about a table in a FROM clause (for JOIN support)
#[derive(Debug, Clone)]
pub struct TableInfo {
    /// Original table name
    pub name: String,
    /// Alias if present, otherwise same as name
    pub effective_name: String,
    /// Whether this table has an explicit alias
    pub has_alias: bool,
}

/// All tables in a query's FROM clause (for JOIN support)
#[derive(Debug, Clone, Default)]
pub struct FromClauseInfo {
    /// All tables including JOINed ones, keyed by effective_name
    pub tables: HashMap<String, TableInfo>,
    /// The primary (first) table
    pub primary_table: Option<TableInfo>,
}

// =============================================================================
// Nom Parsers - Core Building Blocks
// =============================================================================

/// Parse SQL identifier: [a-zA-Z_][a-zA-Z0-9_]*
fn identifier(input: &str) -> IResult<&str, &str> {
    recognize(pair(
        take_while1(|c: char| c.is_alphabetic() || c == '_'),
        take_while(|c: char| c.is_alphanumeric() || c == '_'),
    ))(input)
}

/// Parse content with balanced parentheses, handling quoted strings
fn balanced_parens(input: &str) -> IResult<&str, &str> {
    let start = input;
    let mut depth = 0;
    let mut i = 0;
    let chars: Vec<char> = input.chars().collect();

    while i < chars.len() {
        match chars[i] {
            '(' => depth += 1,
            ')' => {
                if depth == 0 {
                    return Ok((&input[i..], &start[..i]));
                }
                depth -= 1;
            }
            '\'' => {
                // Skip single-quoted string
                i += 1;
                while i < chars.len() && chars[i] != '\'' {
                    if chars[i] == '\\' && i + 1 < chars.len() {
                        i += 1; // Skip escaped char
                    }
                    i += 1;
                }
            }
            '"' => {
                // Skip double-quoted string
                i += 1;
                while i < chars.len() && chars[i] != '"' {
                    if chars[i] == '\\' && i + 1 < chars.len() {
                        i += 1;
                    }
                    i += 1;
                }
            }
            _ => {}
        }
        i += 1;
    }

    // If we exhausted input without finding closing paren at depth 0
    Ok(("", start))
}

/// Parse a function call: IDENTIFIER(content)
fn function_call(input: &str) -> IResult<&str, (&str, &str)> {
    let (input, name) = identifier(input)?;
    let (input, _) = multispace0(input)?;
    let (input, args) = delimited(char('('), balanced_parens, char(')'))(input)?;
    Ok((input, (name, args)))
}

/// Parse an expression or identifier for ad hoc dimensions
/// Handles: `region`, `MONTH(date)`, `YEAR(order_date)`
fn expression_or_identifier(input: &str) -> IResult<&str, String> {
    let (input, name) = identifier(input)?;
    let (input, _) = multispace0(input)?;

    // Check if followed by opening paren (function call)
    if input.starts_with('(') {
        let (input, args) = delimited(char('('), balanced_parens, char(')'))(input)?;
        Ok((input, format!("{name}({args})")))
    } else {
        Ok((input, name.to_string()))
    }
}

// =============================================================================
// Nom Parsers - AS MEASURE
// =============================================================================

/// Check if SQL contains "AS MEASURE" pattern (case-insensitive)
pub fn has_as_measure(sql: &str) -> bool {
    // Quick check using case-insensitive substring matching
    // The actual parsing uses nom, this is just a fast filter
    let sql_upper = sql.to_uppercase();
    sql_upper.contains(" AS MEASURE ")
}

/// Check if SQL contains AGGREGATE( function
pub fn has_aggregate_function(sql: &str) -> bool {
    let chars: Vec<char> = sql.chars().collect();
    let len = chars.len();
    let mut i = 0;

    let is_ident_start = |c: char| c.is_alphabetic() || c == '_';
    let is_ident_char = |c: char| c.is_alphanumeric() || c == '_';

    let skip_whitespace = |mut idx: usize| -> usize {
        while idx < len && chars[idx].is_whitespace() {
            idx += 1;
        }
        idx
    };

    let parse_identifier = |start: usize| -> (String, usize) {
        let mut idx = start + 1;
        while idx < len && is_ident_char(chars[idx]) {
            idx += 1;
        }
        let token: String = chars[start..idx].iter().collect();
        (token, idx)
    };

    let parse_quoted_identifier = |start: usize| -> (String, usize) {
        let mut token = String::new();
        let mut idx = start;
        while idx < len {
            match chars[idx] {
                '"' => {
                    if idx + 1 < len && chars[idx + 1] == '"' {
                        token.push('"');
                        idx += 2;
                    } else {
                        idx += 1;
                        break;
                    }
                }
                c => {
                    token.push(c);
                    idx += 1;
                }
            }
        }
        (token, idx)
    };

    let parse_qualified_chain = |first: String, mut idx: usize| -> (String, usize) {
        let mut last = first;
        loop {
            idx = skip_whitespace(idx);
            if idx >= len || chars[idx] != '.' {
                break;
            }
            idx += 1;
            idx = skip_whitespace(idx);
            if idx >= len {
                break;
            }
            if chars[idx] == '"' {
                let (token, next) = parse_quoted_identifier(idx + 1);
                last = token;
                idx = next;
            } else if is_ident_start(chars[idx]) {
                let (token, next) = parse_identifier(idx);
                last = token;
                idx = next;
            } else {
                break;
            }
        }
        (last, idx)
    };

    let is_aggregate_token = |token: &str| token.eq_ignore_ascii_case("AGGREGATE");

    while i < len {
        match chars[i] {
            '\'' => {
                i += 1;
                while i < len {
                    if chars[i] == '\'' {
                        if i + 1 < len && chars[i + 1] == '\'' {
                            i += 2;
                            continue;
                        }
                        i += 1;
                        break;
                    }
                    i += 1;
                }
            }
            '-' if i + 1 < len && chars[i + 1] == '-' => {
                i += 2;
                while i < len && chars[i] != '\n' {
                    i += 1;
                }
            }
            '/' if i + 1 < len && chars[i + 1] == '*' => {
                i += 2;
                while i + 1 < len {
                    if chars[i] == '*' && chars[i + 1] == '/' {
                        i += 2;
                        break;
                    }
                    i += 1;
                }
            }
            '"' => {
                let (token, next) = parse_quoted_identifier(i + 1);
                let (last, after_chain) = parse_qualified_chain(token, next);
                let after_ws = skip_whitespace(after_chain);
                if after_ws < len && chars[after_ws] == '(' && is_aggregate_token(&last) {
                    return true;
                }
                i = after_chain;
            }
            c if is_ident_start(c) => {
                let (token, next) = parse_identifier(i);
                let (last, after_chain) = parse_qualified_chain(token, next);
                let after_ws = skip_whitespace(after_chain);
                if after_ws < len && chars[after_ws] == '(' && is_aggregate_token(&last) {
                    return true;
                }
                i = after_chain;
            }
            _ => {
                i += 1;
            }
        }
    }

    false
}

fn normalize_identifier_name(name: &str) -> String {
    name.trim()
        .trim_matches('"')
        .trim_matches('`')
        .trim_matches('[')
        .trim_matches(']')
        .to_ascii_lowercase()
}

fn parse_simple_measure_ref(expr: &str) -> Option<(Option<String>, String)> {
    let trimmed = expr.trim();
    if trimmed.is_empty() {
        return None;
    }

    let allowed = |c: char| c.is_ascii_alphanumeric() || c == '_' || c == '.' || c == '"' || c == '`' || c == '[' || c == ']';
    if !trimmed.chars().all(allowed) {
        return None;
    }

    let parts: Vec<&str> = trimmed.split('.').collect();
    match parts.as_slice() {
        [measure] => Some((None, normalize_identifier_name(measure))),
        [qualifier, measure] => Some((
            Some(normalize_identifier_name(qualifier)),
            normalize_identifier_name(measure),
        )),
        _ => None,
    }
}

fn measure_columns_for_query_tables(
    info: &SelectInfo,
) -> (
    HashMap<String, HashSet<String>>,
    HashSet<String>,
) {
    let views = MEASURE_VIEWS.lock().unwrap();
    let mut by_qualifier: HashMap<String, HashSet<String>> = HashMap::new();
    let mut any_measure: HashSet<String> = HashSet::new();

    for table in &info.tables {
        let maybe_view = views.iter().find(|(name, _)| name.eq_ignore_ascii_case(&table.table_name));
        let Some((_, view)) = maybe_view else {
            continue;
        };

        let measures: HashSet<String> = view
            .measures
            .iter()
            .map(|m| normalize_identifier_name(&m.column_name))
            .collect();
        if measures.is_empty() {
            continue;
        }

        any_measure.extend(measures.iter().cloned());

        by_qualifier
            .entry(normalize_identifier_name(&table.table_name))
            .or_default()
            .extend(measures.iter().cloned());

        if let Some(alias) = &table.alias {
            by_qualifier
                .entry(normalize_identifier_name(alias))
                .or_default()
                .extend(measures.iter().cloned());
        }
    }

    (by_qualifier, any_measure)
}

fn select_item_is_implicit_measure_ref(
    item_expr: &str,
    by_qualifier: &HashMap<String, HashSet<String>>,
    any_measure: &HashSet<String>,
) -> bool {
    let Some((qualifier, measure)) = parse_simple_measure_ref(item_expr) else {
        return false;
    };

    if let Some(q) = qualifier {
        return by_qualifier
            .get(&q)
            .map(|names| names.contains(&measure))
            .unwrap_or(false);
    }

    any_measure.contains(&measure)
}

pub fn has_implicit_measure_refs(sql: &str) -> bool {
    let known_measures = known_measure_names();
    if known_measures.is_empty() {
        return false;
    }

    let info = match parser_ffi::parse_select(sql) {
        Ok(info) => info,
        Err(_) => return has_implicit_measure_refs_fallback(sql, &known_measures),
    };

    let (by_qualifier, any_measure) = measure_columns_for_query_tables(&info);
    if any_measure.is_empty() {
        return false;
    }

    info.items
        .iter()
        .any(|item| {
            !item.is_aggregate
                && !item.is_star
                && !item.is_measure_ref
                && select_item_is_implicit_measure_ref(&item.expression_sql, &by_qualifier, &any_measure)
        })
}

fn collect_top_level_select_item_ranges(sql: &str) -> Option<Vec<(usize, usize)>> {
    let query = sql;
    let select_pos = find_top_level_keyword(query, "SELECT", 0)?;
    let from_pos = find_top_level_keyword(query, "FROM", select_pos)?;
    let select_start = select_pos + "SELECT".len();
    if select_start >= from_pos {
        return None;
    }

    let bytes = query.as_bytes();
    let mut ranges = Vec::new();
    let mut item_start = select_start;
    let mut depth = 0i32;
    let mut i = select_start;
    let mut in_single = false;
    let mut in_double = false;

    while i < from_pos {
        let b = bytes[i];
        if in_single {
            if b == b'\'' {
                if i + 1 < from_pos && bytes[i + 1] == b'\'' {
                    i += 2;
                    continue;
                }
                in_single = false;
            }
            i += 1;
            continue;
        }
        if in_double {
            if b == b'"' {
                if i + 1 < from_pos && bytes[i + 1] == b'"' {
                    i += 2;
                    continue;
                }
                in_double = false;
            }
            i += 1;
            continue;
        }

        match b {
            b'\'' => in_single = true,
            b'"' => in_double = true,
            b'(' => depth += 1,
            b')' => {
                if depth > 0 {
                    depth -= 1;
                }
            }
            b',' if depth == 0 => {
                ranges.push((item_start, i));
                item_start = i + 1;
            }
            _ => {}
        }

        i += 1;
    }

    if item_start < from_pos {
        ranges.push((item_start, from_pos));
    }

    Some(ranges)
}

fn find_top_level_as_pos(item: &str) -> Option<usize> {
    let upper = item.to_uppercase();
    let bytes = item.as_bytes();
    let upper_bytes = upper.as_bytes();
    let mut depth = 0i32;
    let mut i = 0usize;
    let mut in_single = false;
    let mut in_double = false;

    while i + 4 <= bytes.len() {
        let b = bytes[i];
        if in_single {
            if b == b'\'' {
                if i + 1 < bytes.len() && bytes[i + 1] == b'\'' {
                    i += 2;
                    continue;
                }
                in_single = false;
            }
            i += 1;
            continue;
        }
        if in_double {
            if b == b'"' {
                if i + 1 < bytes.len() && bytes[i + 1] == b'"' {
                    i += 2;
                    continue;
                }
                in_double = false;
            }
            i += 1;
            continue;
        }

        match b {
            b'\'' => in_single = true,
            b'"' => in_double = true,
            b'(' => depth += 1,
            b')' => {
                if depth > 0 {
                    depth -= 1;
                }
            }
            _ => {}
        }

        if depth == 0 && upper_bytes[i..].starts_with(b" AS ") {
            return Some(i);
        }
        i += 1;
    }

    None
}

fn split_item_expr_and_alias(item: &str) -> (&str, Option<&str>) {
    if let Some(pos) = find_top_level_as_pos(item) {
        let expr = item[..pos].trim();
        let alias = item[pos..].trim();
        (expr, Some(alias))
    } else {
        (item.trim(), None)
    }
}

fn is_simple_known_measure_ref(expr: &str, known_measures: &HashSet<String>) -> bool {
    if has_aggregate_function(expr) {
        return false;
    }
    if expr.to_uppercase().contains(" AT ") {
        return false;
    }
    let Some((_, measure_name)) = parse_simple_measure_ref(expr) else {
        return false;
    };
    known_measures.contains(&measure_name)
}

fn starts_with_top_level_select_or_with(sql: &str) -> bool {
    let trimmed = sql.trim_start();
    let upper = trimmed.to_uppercase();
    upper.starts_with("SELECT ") || upper.starts_with("SELECT\n") || upper.starts_with("WITH ")
}

fn has_implicit_measure_refs_fallback(sql: &str, known_measures: &HashSet<String>) -> bool {
    if !starts_with_top_level_select_or_with(sql) {
        return false;
    }
    let Some(ranges) = collect_top_level_select_item_ranges(sql) else {
        return false;
    };
    ranges.into_iter().any(|(start, end)| {
        let item = sql[start..end].trim();
        if item.is_empty() {
            return false;
        }
        let (expr, _) = split_item_expr_and_alias(item);
        is_simple_known_measure_ref(expr, known_measures)
    })
}

fn rewrite_implicit_measure_refs_fallback(sql: &str, known_measures: &HashSet<String>) -> String {
    if !starts_with_top_level_select_or_with(sql) {
        return sql.to_string();
    }
    let Some(ranges) = collect_top_level_select_item_ranges(sql) else {
        return sql.to_string();
    };

    let mut replacements: Vec<(usize, usize, String)> = Vec::new();
    for (start, end) in ranges {
        let raw_item = &sql[start..end];
        let item = raw_item.trim();
        if item.is_empty() {
            continue;
        }

        let (expr, alias_clause) = split_item_expr_and_alias(item);
        if !is_simple_known_measure_ref(expr, known_measures) {
            continue;
        }

        let mut rewritten = format!("AGGREGATE({}) {}", expr.trim(), DEFAULT_CONTEXT_MARKER);
        if let Some(alias) = alias_clause {
            rewritten.push(' ');
            rewritten.push_str(alias);
        }
        // Preserve token separation before the next clause keyword (e.g. FROM).
        rewritten.push(' ');
        replacements.push((start, end, rewritten));
    }

    if replacements.is_empty() {
        return sql.to_string();
    }

    let mut result = sql.to_string();
    replacements.sort_by(|a, b| b.0.cmp(&a.0));
    for (start, end, rewritten) in replacements {
        if start <= end && end <= result.len() {
            result.replace_range(start..end, &rewritten);
        }
    }
    result
}

fn rewrite_implicit_measure_refs(sql: &str) -> String {
    let known_measures = known_measure_names();
    if known_measures.is_empty() {
        return sql.to_string();
    }

    let info = match parser_ffi::parse_select(sql) {
        Ok(info) => info,
        Err(_) => return rewrite_implicit_measure_refs_fallback(sql, &known_measures),
    };

    let (by_qualifier, any_measure) = measure_columns_for_query_tables(&info);
    if any_measure.is_empty() {
        return rewrite_implicit_measure_refs_fallback(sql, &known_measures);
    }

    let mut replacements: Vec<parser_ffi::Replacement> = Vec::new();
    for item in &info.items {
        if item.is_aggregate || item.is_star || item.is_measure_ref {
            continue;
        }
        if !select_item_is_implicit_measure_ref(&item.expression_sql, &by_qualifier, &any_measure) {
            continue;
        }

        // Plain measure refs follow paper default semantics (not VISIBLE).
        // Keep an internal marker so expansion can apply default context rules.
        let mut rewritten = format!(
            "AGGREGATE({}) {}",
            item.expression_sql.trim(),
            DEFAULT_CONTEXT_MARKER
        );
        if let Some(alias) = &item.alias {
            rewritten.push_str(" AS ");
            rewritten.push_str(alias);
        }
        // Ensure separator before the next token (e.g. FROM) when replacing final SELECT item.
        rewritten.push(' ');

        replacements.push(parser_ffi::Replacement {
            start_pos: item.start_pos,
            end_pos: item.end_pos,
            replacement: rewritten,
        });
    }

    if replacements.is_empty() {
        return sql.to_string();
    }

    parser_ffi::apply_replacements(sql, &replacements).unwrap_or_else(|_| sql.to_string())
}

fn known_measure_names() -> HashSet<String> {
    let views = MEASURE_VIEWS.lock().unwrap();
    views
        .values()
        .flat_map(|view| view.measures.iter().map(|m| normalize_identifier_name(&m.column_name)))
        .collect()
}

fn is_measure_ref_char(c: u8) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, b'_' | b'.' | b'"' | b'`' | b'[' | b']')
}

fn is_keyword_boundary(bytes: &[u8], start: usize, end: usize) -> bool {
    let left_ok = start == 0 || !(bytes[start - 1].is_ascii_alphanumeric() || bytes[start - 1] == b'_');
    let right_ok = end >= bytes.len() || !(bytes[end].is_ascii_alphanumeric() || bytes[end] == b'_');
    left_ok && right_ok
}

fn find_previous_measure_ref_bounds(sql: &str, at_start: usize) -> Option<(usize, usize)> {
    let bytes = sql.as_bytes();
    let mut end = at_start;
    while end > 0 && bytes[end - 1].is_ascii_whitespace() {
        end -= 1;
    }
    if end == 0 || bytes[end - 1] == b')' {
        return None;
    }

    let mut start = end;
    while start > 0 && is_measure_ref_char(bytes[start - 1]) {
        start -= 1;
    }
    if start == end {
        return None;
    }

    if start > 0 {
        let prev = bytes[start - 1];
        if is_measure_ref_char(prev) || prev == b')' {
            return None;
        }
    }

    Some((start, end))
}

fn rewrite_measure_at_refs(sql: &str) -> String {
    let known_measures = known_measure_names();
    if known_measures.is_empty() {
        return sql.to_string();
    }

    let bytes = sql.as_bytes();
    let upper = sql.to_uppercase();
    let upper_bytes = upper.as_bytes();
    let mut replacements: Vec<parser_ffi::Replacement> = Vec::new();
    let mut i = 0usize;

    while i < bytes.len() {
        match bytes[i] {
            b'\'' => {
                i += 1;
                while i < bytes.len() {
                    if bytes[i] == b'\'' {
                        if i + 1 < bytes.len() && bytes[i + 1] == b'\'' {
                            i += 2;
                            continue;
                        }
                        i += 1;
                        break;
                    }
                    i += 1;
                }
                continue;
            }
            b'"' => {
                i += 1;
                while i < bytes.len() {
                    if bytes[i] == b'"' {
                        if i + 1 < bytes.len() && bytes[i + 1] == b'"' {
                            i += 2;
                            continue;
                        }
                        i += 1;
                        break;
                    }
                    i += 1;
                }
                continue;
            }
            b'-' if i + 1 < bytes.len() && bytes[i + 1] == b'-' => {
                i += 2;
                while i < bytes.len() && bytes[i] != b'\n' {
                    i += 1;
                }
                continue;
            }
            b'/' if i + 1 < bytes.len() && bytes[i + 1] == b'*' => {
                i += 2;
                while i + 1 < bytes.len() {
                    if bytes[i] == b'*' && bytes[i + 1] == b'/' {
                        i += 2;
                        break;
                    }
                    i += 1;
                }
                continue;
            }
            _ => {}
        }

        if i + 2 <= bytes.len()
            && upper_bytes[i..].starts_with(b"AT")
            && is_keyword_boundary(bytes, i, i + 2)
        {
            let mut j = i + 2;
            while j < bytes.len() && bytes[j].is_ascii_whitespace() {
                j += 1;
            }
            if j < bytes.len() && bytes[j] == b'(' {
                if let Some((start, end)) = find_previous_measure_ref_bounds(sql, i) {
                    let token = sql[start..end].trim();
                    if let Some((_, measure_name)) = parse_simple_measure_ref(token) {
                        if known_measures.contains(&measure_name) {
                            replacements.push(parser_ffi::Replacement {
                                start_pos: start as u32,
                                end_pos: end as u32,
                                replacement: format!("AGGREGATE({token})"),
                            });
                        }
                    }
                }
            }
            i += 2;
            continue;
        }

        i += 1;
    }

    if replacements.is_empty() {
        return sql.to_string();
    }

    parser_ffi::apply_replacements(sql, &replacements).unwrap_or_else(|_| sql.to_string())
}

pub fn has_measure_at_refs(sql: &str) -> bool {
    rewrite_measure_at_refs(sql) != sql
}

fn strip_measure_qualifier(measure: &str) -> String {
    measure
        .trim()
        .split('.')
        .next_back()
        .unwrap_or(measure.trim())
        .trim()
        .trim_matches('"')
        .trim_matches('`')
        .trim_matches('[')
        .trim_matches(']')
        .to_string()
}

/// Check if SQL contains curly brace measure syntax: `{column}`
pub fn has_curly_brace_measure(sql: &str) -> bool {
    let mut parser = delimited(
        char::<_, nom::error::Error<&str>>('{'),
        take_while1(|c: char| c.is_alphanumeric() || c == '_'),
        char('}'),
    );

    let mut remaining = sql;
    while !remaining.is_empty() {
        if parser(remaining).is_ok() {
            return true;
        }
        if remaining.len() > 1 {
            remaining = &remaining[1..];
        } else {
            break;
        }
    }
    false
}

/// Convert curly brace syntax `{column}` to `AGGREGATE(column)`
pub fn expand_curly_braces(sql: &str) -> String {
    let mut result = String::new();
    let mut chars = sql.chars().peekable();

    while let Some(c) = chars.next() {
        if c == '{' {
            let mut ident = String::new();
            while let Some(&next) = chars.peek() {
                if next == '}' {
                    chars.next();
                    break;
                }
                ident.push(chars.next().unwrap());
            }
            if !ident.is_empty() {
                result.push_str(&format!("AGGREGATE({ident})"));
            } else {
                result.push('{');
            }
        } else {
            result.push(c);
        }
    }
    result
}

// =============================================================================
// Nom Parsers - AT Syntax
// =============================================================================

/// Parse AT (ALL) with no dimension - grand total
fn at_all_global(input: &str) -> IResult<&str, ContextModifier> {
    let (input, _) = tag_no_case("ALL")(input)?;
    // Must be followed by ) or whitespace then ), or end of input, not an identifier
    let (input, _) = multispace0(input)?;
    // Succeed if at end of input OR at closing paren
    if input.is_empty() || input.starts_with(')') {
        Ok((input, ContextModifier::AllGlobal))
    } else {
        Err(nom::Err::Error(nom::error::Error::new(
            input,
            nom::error::ErrorKind::Char,
        )))
    }
}

/// Parse AT (ALL dimension) - remove dimension from context
/// Supports ad hoc dimensions: `ALL region` or `ALL MONTH(date)`
fn at_all(input: &str) -> IResult<&str, ContextModifier> {
    let (input, _) = tag_no_case("ALL")(input)?;
    let (input, _) = multispace1(input)?;
    let (input, dim) = expression_or_identifier(input)?;
    Ok((input, ContextModifier::All(dim)))
}

/// Parse AT (VISIBLE) - respect outer query's WHERE clause
fn at_visible(input: &str) -> IResult<&str, ContextModifier> {
    let (input, _) = tag_no_case("VISIBLE")(input)?;
    Ok((input, ContextModifier::Visible))
}

/// Parse CURRENT dimension reference in expressions
/// CURRENT is resolved later once GROUP BY/WHERE context is known.
fn parse_current_in_expr(expr: &str) -> String {
    expr.trim().to_string()
}

/// Parse AT (SET dimension = expr)
/// Supports ad hoc dimensions: `SET year = 2023` or `SET MONTH(date) = 3`
fn at_set(input: &str) -> IResult<&str, ContextModifier> {
    let (input, _) = tag_no_case("SET")(input)?;
    let (input, _) = multispace1(input)?;
    let (input, dim) = expression_or_identifier(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char('=')(input)?;
    let (input, _) = multispace0(input)?;
    // Take rest as expression (until closing paren, handled by caller)
    let (input, expr) = take_while(|c: char| c != ')')(input)?;
    // Process CURRENT references in the expression
    let processed_expr = parse_current_in_expr(expr.trim());
    Ok((input, ContextModifier::Set(dim, processed_expr)))
}

/// Parse AT (WHERE condition)
fn at_where(input: &str) -> IResult<&str, ContextModifier> {
    let (input, _) = tag_no_case("WHERE")(input)?;
    let (input, _) = multispace1(input)?;
    // AT (...) content is already balanced by the caller, so WHERE can
    // safely consume the remainder here (including nested function parens).
    let cond = input.trim();
    let stripped = strip_at_where_qualifiers(cond.trim());
    Ok(("", ContextModifier::Where(stripped)))
}

/// Parse any AT modifier content
fn at_modifier(input: &str) -> IResult<&str, ContextModifier> {
    let (input, _) = multispace0(input)?;
    // Order matters: at_all_global must come before at_all since both start with "ALL"
    alt((at_all_global, at_all, at_set, at_where, at_visible))(input)
}

/// Parse AT modifier from string (public API)
pub fn parse_at_modifier(content: &str) -> Result<ContextModifier> {
    at_modifier(content)
        .map(|(_, m)| m)
        .map_err(|e| YardstickError::Validation(format!("Failed to parse AT modifier: {e:?}")))
}

/// Check if SQL contains AT syntax using nom
pub fn has_at_syntax(sql: &str) -> bool {
    // Look for pattern: ) AT (
    let mut pattern = tuple((
        char::<_, nom::error::Error<&str>>(')'),
        multispace0,
        tag_no_case("AT"),
        multispace0,
        char('('),
    ));

    let mut remaining = sql;
    while !remaining.is_empty() {
        if pattern(remaining).is_ok() {
            return true;
        }
        if remaining.len() > 1 {
            remaining = &remaining[1..];
        } else {
            break;
        }
    }
    false
}

/// Parsed AGGREGATE with optional AT
#[derive(Debug)]
pub struct AggregateCall {
    pub measure_name: String,
    pub at_modifier: Option<ContextModifier>,
    pub start_pos: usize,
    pub end_pos: usize,
}

/// Parse multiple modifiers inside a single AT clause (e.g., AT (SET year = year - 1 VISIBLE))
fn at_modifiers_content(input: &str) -> IResult<&str, Vec<ContextModifier>> {
    fn starts_with_modifier_keyword(s: &str) -> bool {
        let trimmed = s.trim_start();
        let upper = trimmed.to_uppercase();
        upper.starts_with("ALL ")
            || upper == "ALL"
            || upper.starts_with("SET ")
            || upper.starts_with("WHERE ")
            || upper == "VISIBLE"
            || upper.starts_with("VISIBLE ")
    }

    let mut modifiers = Vec::new();
    let mut remaining = input.trim();

    // Keep parsing modifiers until we can't parse any more
    while !remaining.is_empty() {
        let result = at_modifier(remaining);
        match result {
            Ok((rest, modifier)) => {
                modifiers.push(modifier);
                remaining = rest.trim();

                if let Some(ContextModifier::All(_)) = modifiers.last() {
                    while !remaining.is_empty() && !starts_with_modifier_keyword(remaining) {
                        if let Ok((after_dim, dim)) = expression_or_identifier(remaining) {
                            modifiers.push(ContextModifier::All(dim));
                            remaining = after_dim.trim();
                        } else {
                            break;
                        }
                    }
                }
            }
            Err(_) => break,
        }
    }

    if modifiers.is_empty() {
        Err(nom::Err::Error(nom::error::Error::new(
            input,
            nom::error::ErrorKind::Tag,
        )))
    } else {
        Ok((remaining, modifiers))
    }
}

/// Parse AGGREGATE(measure) optionally followed by one or more AT (modifier) clauses
/// Returns Vec<ContextModifier> to support both chaining and multiple modifiers in single AT
fn aggregate_with_at(input: &str) -> IResult<&str, (&str, Vec<ContextModifier>)> {
    let (input, _) = tag_no_case("AGGREGATE")(input)?;
    let (input, _) = multispace0(input)?;
    let (input, measure) = delimited(char('('), balanced_parens, char(')'))(input)?;

    // Collect all AT modifiers (for chaining and multi-modifier AT clauses)
    let mut all_modifiers = Vec::new();
    let mut remaining = input;

    loop {
        // Try to parse AT (...)
        let at_start: IResult<&str, _> =
            tuple((multispace0, tag_no_case("AT"), multispace0, char('(')))(remaining);

        match at_start {
            Ok((after_open, _)) => {
                // Find the matching close paren
                if let Ok((after_content, content)) = balanced_parens(after_open) {
                    // Skip the closing paren
                    if let Ok((after_close, _)) =
                        char::<_, nom::error::Error<&str>>(')')(after_content)
                    {
                        // Parse modifiers from content
                        if let Ok((_, mods)) = at_modifiers_content(content) {
                            all_modifiers.extend(mods);
                        }
                        remaining = after_close;
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }
            Err(_) => break,
        }
    }

    Ok((remaining, (measure.trim(), all_modifiers)))
}

/// Extract all AGGREGATE(...) [AT (...)]+ patterns from SQL with full modifier list
pub fn extract_aggregate_with_at_full(
    sql: &str,
) -> Vec<(String, Vec<ContextModifier>, usize, usize)> {
    let mut results = Vec::new();
    let sql_upper = sql.to_uppercase();
    let mut search_pos = 0;

    while let Some(agg_offset) = sql_upper[search_pos..].find("AGGREGATE") {
        let start = search_pos + agg_offset;

        if let Ok((remaining, (measure, modifiers))) = aggregate_with_at(&sql[start..]) {
            if !modifiers.is_empty() {
                let end = sql.len() - remaining.len();
                results.push((measure.to_string(), modifiers, start, end));
            }
        }

        search_pos = start + 1;
    }

    results
}

/// Extract all AGGREGATE(...) [AT (...)] patterns from SQL (legacy - returns first modifier)
pub fn extract_aggregate_with_at(sql: &str) -> Vec<(String, ContextModifier, usize, usize)> {
    extract_aggregate_with_at_full(sql)
        .into_iter()
        .map(|(measure, mut modifiers, start, end)| {
            // For backward compatibility, combine modifiers
            let modifier = if modifiers.len() == 1 {
                modifiers.remove(0)
            } else {
                // Check if all are ALL modifiers
                let all_are_all = modifiers
                    .iter()
                    .all(|m| matches!(m, ContextModifier::All(_) | ContextModifier::AllGlobal));
                if all_are_all {
                    ContextModifier::AllGlobal
                } else {
                    modifiers.remove(0)
                }
            };
            (measure, modifier, start, end)
        })
        .collect()
}

/// Extract all AGGREGATE() calls without AT
pub fn extract_all_aggregate_calls(sql: &str) -> Vec<(String, usize, usize)> {
    let mut results = Vec::new();
    let sql_upper = sql.to_uppercase();
    let mut search_pos = 0;

    while let Some(agg_offset) = sql_upper[search_pos..].find("AGGREGATE") {
        let start = search_pos + agg_offset;

        if let Ok((remaining, (measure, modifiers))) = aggregate_with_at(&sql[start..]) {
            // Only include calls WITHOUT AT modifier
            if modifiers.is_empty() {
                let end = sql.len() - remaining.len();
                results.push((measure.to_string(), start, end));
            }
        }

        search_pos = start + 1;
    }

    results
}

// =============================================================================
// Nom Parsers - CREATE VIEW
// =============================================================================

/// Parse CREATE [OR REPLACE] VIEW name
fn create_view_header(input: &str) -> IResult<&str, &str> {
    let (input, _) = multispace0(input)?;
    let (input, _) = tag_no_case("CREATE")(input)?;
    let (input, _) = multispace1(input)?;

    // Optional OR REPLACE
    let (input, _) = opt(tuple((
        tag_no_case("OR"),
        multispace1,
        tag_no_case("REPLACE"),
        multispace1,
    )))(input)?;

    let (input, _) = tag_no_case("VIEW")(input)?;
    let (input, _) = multispace1(input)?;
    let (input, name) = identifier(input)?;

    Ok((input, name))
}

/// Extract view name from CREATE VIEW statement
pub fn extract_view_name(sql: &str) -> Option<String> {
    create_view_header(sql)
        .ok()
        .map(|(_, name)| name.to_string())
}

/// Extract view name from DROP VIEW statement
pub fn extract_drop_view_name(sql: &str) -> Option<String> {
    let upper = sql.to_uppercase();
    let mut idx = skip_ws_and_comments(sql, 0);
    if !matches_keyword_at(&upper, idx, "DROP") {
        return None;
    }
    idx += "DROP".len();
    idx = skip_ws_and_comments(sql, idx);

    if !matches_keyword_at(&upper, idx, "VIEW") {
        return None;
    }
    idx += "VIEW".len();
    idx = skip_ws_and_comments(sql, idx);

    if matches_keyword_at(&upper, idx, "IF") {
        idx += "IF".len();
        idx = skip_ws_and_comments(sql, idx);
        if !matches_keyword_at(&upper, idx, "EXISTS") {
            return None;
        }
        idx += "EXISTS".len();
        idx = skip_ws_and_comments(sql, idx);
    }

    let end = parse_qualified_name_span(sql, idx)?;
    if !is_statement_tail(sql, end) {
        return None;
    }
    extract_last_qualified_identifier(&sql[idx..end])
}

/// Extract table name from SQL FROM clause
pub fn extract_table_name_from_sql(sql: &str) -> Option<String> {
    extract_table_and_alias_from_sql(sql).map(|(name, _)| name)
}

/// Extract table name and optional alias from SQL FROM clause
/// Returns (table_name, Option<alias>)
pub fn extract_table_and_alias_from_sql(sql: &str) -> Option<(String, Option<String>)> {
    let from_pos = find_top_level_keyword(sql, "FROM", 0)?;
    let upper = sql.to_uppercase();
    let i = skip_ws_and_comments(sql, from_pos + 4);
    let bytes = sql.as_bytes();
    if i >= bytes.len() || bytes[i] == b'(' {
        return None;
    }

    let table_end = parse_qualified_name_span(sql, i)?;
    let raw_table = sql[i..table_end].trim();
    let table_name =
        extract_last_qualified_identifier(raw_table).unwrap_or_else(|| raw_table.to_string());

    let j = skip_ws_and_comments(sql, table_end);
    if j >= bytes.len() || sql[j..].starts_with(';') {
        return Some((table_name, None));
    }

    if matches_keyword_at(&upper, j, "WHERE")
        || matches_keyword_at(&upper, j, "GROUP")
        || matches_keyword_at(&upper, j, "ORDER")
        || matches_keyword_at(&upper, j, "LIMIT")
        || matches_keyword_at(&upper, j, "HAVING")
        || matches_keyword_at(&upper, j, "JOIN")
    {
        return Some((table_name, None));
    }

    if let Some((_, alias)) = parse_alias_span(sql, j) {
        Some((table_name, Some(alias)))
    } else {
        Some((table_name, None))
    }
}

fn insert_primary_table_alias(sql: &str, alias: &str) -> Option<String> {
    // Cases:
    // - no top-level FROM: leave unchanged
    // - FROM starts with subquery: leave unchanged
    // - existing alias (with or without AS, quoted or unquoted): leave unchanged
    // - no alias: insert the provided alias after the primary table name
    let from_pos = find_top_level_keyword(sql, "FROM", 0)?;
    let bytes = sql.as_bytes();
    let i = skip_ws_and_comments(sql, from_pos + 4);
    if i >= bytes.len() || bytes[i] == b'(' {
        return None;
    }

    let table_end = parse_qualified_name_span(sql, i)?;
    let j = skip_ws_and_comments(sql, table_end);
    if parse_alias_span(sql, j).is_some() {
        return None;
    }

    let mut result = String::with_capacity(sql.len() + alias.len() + 1);
    result.push_str(&sql[..table_end]);
    result.push(' ');
    result.push_str(alias);
    result.push_str(&sql[table_end..]);
    Some(result)
}

/// Extract the SELECT/WITH query from a CREATE VIEW statement
fn extract_view_query(sql: &str) -> Option<String> {
    let trimmed = sql.trim();
    let upper = trimmed.to_uppercase();
    if upper.starts_with("SELECT ") || upper.starts_with("WITH ") {
        return Some(trimmed.to_string());
    }

    let (rest, _) = create_view_header(trimmed).ok()?;
    let rest_trimmed = rest.trim_start();
    let rest_upper = rest_trimmed.to_uppercase();

    if rest_upper.starts_with("AS ") {
        return Some(rest_trimmed[2..].trim_start().to_string());
    }

    if rest_upper.starts_with("AS") && rest_trimmed.len() > 2 {
        let after_as = &rest_trimmed[2..];
        if after_as
            .chars()
            .next()
            .map_or(false, |ch| ch.is_whitespace())
        {
            return Some(after_as.trim_start().to_string());
        }
    }

    if rest_upper.starts_with("SELECT") || rest_upper.starts_with("WITH") {
        return Some(rest_trimmed.to_string());
    }

    None
}

fn is_boundary_char(ch: Option<char>) -> bool {
    ch.map_or(true, |c| !c.is_alphanumeric() && c != '_')
}

fn is_statement_tail(sql: &str, mut idx: usize) -> bool {
    let bytes = sql.as_bytes();
    idx = skip_ws_and_comments(sql, idx);
    while idx < bytes.len() && bytes[idx] == b';' {
        idx += 1;
        idx = skip_ws_and_comments(sql, idx);
    }
    idx == bytes.len()
}

fn skip_ws_and_comments(sql: &str, mut idx: usize) -> usize {
    let bytes = sql.as_bytes();
    while idx < bytes.len() {
        if bytes[idx].is_ascii_whitespace() {
            idx += 1;
            continue;
        }
        if bytes[idx] == b'-' && idx + 1 < bytes.len() && bytes[idx + 1] == b'-' {
            idx += 2;
            while idx < bytes.len() {
                let ch = bytes[idx];
                idx += 1;
                if ch == b'\n' || ch == b'\r' {
                    break;
                }
            }
            continue;
        }
        if bytes[idx] == b'/' && idx + 1 < bytes.len() && bytes[idx + 1] == b'*' {
            idx += 2;
            while idx + 1 < bytes.len() {
                if bytes[idx] == b'*' && bytes[idx + 1] == b'/' {
                    idx += 2;
                    break;
                }
                idx += 1;
            }
            if idx + 1 >= bytes.len() {
                idx = bytes.len();
            }
            continue;
        }
        break;
    }
    idx
}

fn is_ident_start_byte(b: u8) -> bool {
    (b as char).is_ascii_alphabetic() || b == b'_'
}

fn is_ident_part_byte(b: u8) -> bool {
    (b as char).is_ascii_alphanumeric() || b == b'_'
}

fn strip_identifier_quotes(input: &str) -> &str {
    if input.len() >= 2 {
        let bytes = input.as_bytes();
        let (first, last) = (bytes[0], bytes[bytes.len() - 1]);
        if (first == b'"' && last == b'"')
            || (first == b'`' && last == b'`')
            || (first == b'[' && last == b']')
        {
            return &input[1..input.len() - 1];
        }
    }
    input
}

fn extract_last_qualified_identifier(input: &str) -> Option<String> {
    let bytes = input.as_bytes();
    let mut i = skip_ws_and_comments(input, 0);
    let (end, _) = parse_identifier_token(input, i)?;
    let mut last_start = i;
    let mut last_end = end;
    i = end;

    loop {
        let mut j = skip_ws_and_comments(input, i);
        if j >= bytes.len() || bytes[j] != b'.' {
            break;
        }
        j += 1;
        j = skip_ws_and_comments(input, j);
        let (next_end, _) = parse_identifier_token(input, j)?;
        last_start = j;
        last_end = next_end;
        i = next_end;
    }

    let tail = skip_ws_and_comments(input, i);
    if tail < bytes.len() {
        return None;
    }

    Some(strip_identifier_quotes(input[last_start..last_end].trim()).to_string())
}

fn parse_identifier_token(sql: &str, start: usize) -> Option<(usize, bool)> {
    let bytes = sql.as_bytes();
    if start >= bytes.len() {
        return None;
    }
    let first = bytes[start];
    if first == b'"' || first == b'`' || first == b'[' {
        let end_char = match first {
            b'"' => b'"',
            b'`' => b'`',
            _ => b']',
        };
        let mut i = start + 1;
        while i < bytes.len() {
            if bytes[i] == end_char {
                if end_char == b'"' && i + 1 < bytes.len() && bytes[i + 1] == b'"' {
                    i += 2;
                    continue;
                }
                return Some((i + 1, true));
            }
            i += 1;
        }
        return None;
    }
    if !is_ident_start_byte(first) {
        return None;
    }
    let mut i = start + 1;
    while i < bytes.len() && is_ident_part_byte(bytes[i]) {
        i += 1;
    }
    Some((i, false))
}

fn parse_qualified_name_span(sql: &str, start: usize) -> Option<usize> {
    let bytes = sql.as_bytes();
    let mut i = start;
    let (mut end, _) = parse_identifier_token(sql, i)?;
    i = end;

    loop {
        let mut j = skip_ws_and_comments(sql, i);
        if j >= bytes.len() || bytes[j] != b'.' {
            break;
        }
        j += 1;
        j = skip_ws_and_comments(sql, j);
        let (next_end, _) = parse_identifier_token(sql, j)?;
        end = next_end;
        i = end;
    }

    Some(end)
}

fn parse_alias_span(sql: &str, start: usize) -> Option<(usize, String)> {
    let upper = sql.to_uppercase();
    let mut i = skip_ws_and_comments(sql, start);
    let bytes = sql.as_bytes();
    if i >= bytes.len() {
        return None;
    }

    if matches_keyword_at(&upper, i, "AS") {
        i += 2;
        i = skip_ws_and_comments(sql, i);
    }

    let (end, quoted) = parse_identifier_token(sql, i)?;
    let raw = sql[i..end].trim().to_string();
    if quoted {
        return Some((end, raw));
    }

    let alias_upper = raw.to_uppercase();
    if matches!(
        alias_upper.as_str(),
        "WHERE"
            | "GROUP"
            | "ORDER"
            | "LIMIT"
            | "HAVING"
            | "JOIN"
            | "ON"
            | "USING"
            | "INNER"
            | "LEFT"
            | "RIGHT"
            | "FULL"
            | "CROSS"
            | "UNION"
            | "EXCEPT"
            | "INTERSECT"
            | "WINDOW"
            | "QUALIFY"
    ) {
        return None;
    }

    Some((end, raw))
}

fn find_top_level_keyword(sql: &str, keyword: &str, start: usize) -> Option<usize> {
    let upper = sql.to_uppercase();
    let upper_bytes = upper.as_bytes();
    let keyword_upper = keyword.to_uppercase();
    let keyword_parts: Vec<&str> = keyword_upper.split_whitespace().collect();
    if keyword_parts.is_empty() {
        return None;
    }
    let keyword_bytes = keyword_upper.as_bytes();
    let is_multi_word = keyword_parts.len() > 1;

    let bytes = sql.as_bytes();
    let mut depth: i32 = 0;
    let mut in_single = false;
    let mut in_double = false;
    let mut in_backtick = false;
    let mut in_bracket = false;
    let mut in_line_comment = false;
    let mut in_block_comment = false;

    let mut i = start;
    while i < bytes.len() {
        let c = bytes[i] as char;

        if in_line_comment {
            if c == '\n' || c == '\r' {
                in_line_comment = false;
            }
            i += 1;
            continue;
        }

        if in_block_comment {
            if c == '*' && i + 1 < bytes.len() && bytes[i + 1] as char == '/' {
                in_block_comment = false;
                i += 2;
                continue;
            }
            i += 1;
            continue;
        }

        if in_single {
            if c == '\'' {
                if i + 1 < bytes.len() && bytes[i + 1] as char == '\'' {
                    i += 1;
                } else {
                    in_single = false;
                }
            }
            i += 1;
            continue;
        }

        if in_double {
            if c == '"' {
                in_double = false;
            }
            i += 1;
            continue;
        }

        if in_backtick {
            if c == '`' {
                in_backtick = false;
            }
            i += 1;
            continue;
        }

        if in_bracket {
            if c == ']' {
                in_bracket = false;
            }
            i += 1;
            continue;
        }

        if c == '-' && i + 1 < bytes.len() && bytes[i + 1] as char == '-' {
            in_line_comment = true;
            i += 2;
            continue;
        }

        if c == '/' && i + 1 < bytes.len() && bytes[i + 1] as char == '*' {
            in_block_comment = true;
            i += 2;
            continue;
        }

        match c {
            '\'' => {
                in_single = true;
                i += 1;
                continue;
            }
            '"' => {
                in_double = true;
                i += 1;
                continue;
            }
            '`' => {
                in_backtick = true;
                i += 1;
                continue;
            }
            '[' => {
                in_bracket = true;
                i += 1;
                continue;
            }
            '(' => {
                depth += 1;
                i += 1;
                continue;
            }
            ')' => {
                depth -= 1;
                i += 1;
                continue;
            }
            _ => {}
        }

        if depth == 0 {
            if is_multi_word {
                let mut idx = i;
                let mut matched = true;
                for (part_idx, part) in keyword_parts.iter().enumerate() {
                    let part_bytes = part.as_bytes();
                    if idx + part_bytes.len() > upper_bytes.len()
                        || upper_bytes[idx..idx + part_bytes.len()] != *part_bytes
                    {
                        matched = false;
                        break;
                    }
                    idx += part_bytes.len();
                    if part_idx < keyword_parts.len() - 1 {
                        if idx >= upper_bytes.len()
                            || !upper_bytes[idx].is_ascii_whitespace()
                        {
                            matched = false;
                            break;
                        }
                        while idx < upper_bytes.len() && upper_bytes[idx].is_ascii_whitespace() {
                            idx += 1;
                        }
                    }
                }

                if matched {
                    let prev = if i == 0 {
                        None
                    } else {
                        Some(upper_bytes[i - 1] as char)
                    };
                    let next = if idx >= upper_bytes.len() {
                        None
                    } else {
                        Some(upper_bytes[idx] as char)
                    };
                    if is_boundary_char(prev) && is_boundary_char(next) {
                        return Some(i);
                    }
                }
            } else if i + keyword_bytes.len() <= upper_bytes.len()
                && upper_bytes[i..i + keyword_bytes.len()] == *keyword_bytes
            {
                let prev = if i == 0 {
                    None
                } else {
                    Some(upper_bytes[i - 1] as char)
                };
                let next = if i + keyword_bytes.len() >= upper_bytes.len() {
                    None
                } else {
                    Some(upper_bytes[i + keyword_bytes.len()] as char)
                };
                if is_boundary_char(prev) && is_boundary_char(next) {
                    return Some(i);
                }
            }
        }

        i += 1;
    }

    None
}

fn find_first_top_level_keyword(sql: &str, start: usize, keywords: &[&str]) -> Option<usize> {
    keywords
        .iter()
        .filter_map(|kw| find_top_level_keyword(sql, kw, start))
        .min()
}

fn has_top_level_group_by(sql: &str) -> bool {
    find_top_level_keyword(sql, "GROUP BY", 0).is_some()
}

fn has_group_by_anywhere(sql: &str) -> bool {
    let upper = sql.to_uppercase();
    let mut idx = 0;
    while idx < upper.len() {
        if matches_keyword_at(&upper, idx, "GROUP") {
            let mut next = idx + "GROUP".len();
            next = skip_whitespace(sql, next);
            if matches_keyword_at(&upper, next, "BY") {
                return true;
            }
        }
        idx += 1;
    }
    false
}

fn skip_whitespace(sql: &str, mut idx: usize) -> usize {
    while idx < sql.len() && sql.as_bytes()[idx].is_ascii_whitespace() {
        idx += 1;
    }
    idx
}

fn matches_keyword_at(upper: &str, idx: usize, keyword: &str) -> bool {
    if idx + keyword.len() > upper.len() {
        return false;
    }
    if &upper[idx..idx + keyword.len()] != keyword {
        return false;
    }

    let prev = if idx == 0 {
        None
    } else {
        upper[..idx].chars().rev().next()
    };
    let next = if idx + keyword.len() >= upper.len() {
        None
    } else {
        upper[idx + keyword.len()..].chars().next()
    };

    is_boundary_char(prev) && is_boundary_char(next)
}

fn advance_after_group_by(query: &str, group_pos: usize) -> Option<usize> {
    let upper = query.to_uppercase();
    let mut idx = group_pos;
    if !matches_keyword_at(&upper, idx, "GROUP") {
        return None;
    }
    idx += "GROUP".len();
    idx = skip_whitespace(query, idx);
    if !matches_keyword_at(&upper, idx, "BY") {
        return None;
    }
    idx += "BY".len();
    Some(skip_whitespace(query, idx))
}

struct CteExpansion {
    sql: String,
    had_aggregate: bool,
}

fn expand_cte_queries(sql: &str) -> CteExpansion {
    let with_pos = match find_top_level_keyword(sql, "WITH", 0) {
        Some(pos) => pos,
        None => {
            return CteExpansion {
                sql: sql.to_string(),
                had_aggregate: false,
            };
        }
    };

    let upper = sql.to_uppercase();
    let mut idx = with_pos + "WITH".len();
    let mut replacements: Vec<(usize, usize, String)> = Vec::new();
    let mut had_aggregate = false;

    idx = skip_whitespace(sql, idx);
    if matches_keyword_at(&upper, idx, "RECURSIVE") {
        idx += "RECURSIVE".len();
    }

    loop {
        idx = skip_whitespace(sql, idx);
        if idx >= sql.len() {
            return CteExpansion {
                sql: sql.to_string(),
                had_aggregate: false,
            };
        }

        let rest = &sql[idx..];
        let (after_ident, _) = match identifier(rest) {
            Ok(value) => value,
            Err(_) => {
                return CteExpansion {
                    sql: sql.to_string(),
                    had_aggregate: false,
                };
            }
        };
        idx += rest.len() - after_ident.len();

        idx = skip_whitespace(sql, idx);
        if idx < sql.len() && sql.as_bytes()[idx] == b'(' {
            let sub = &sql[idx + 1..];
            let (rest_after_cols, _) = match balanced_parens(sub) {
                Ok(value) => value,
                Err(_) => {
                    return CteExpansion {
                        sql: sql.to_string(),
                        had_aggregate: false,
                    };
                }
            };
            let cols_len = sub.len() - rest_after_cols.len();
            idx = idx + 1 + cols_len + 1;
            idx = skip_whitespace(sql, idx);
        }

        if !matches_keyword_at(&upper, idx, "AS") {
            return CteExpansion {
                sql: sql.to_string(),
                had_aggregate: false,
            };
        }
        idx += "AS".len();
        idx = skip_whitespace(sql, idx);

        if idx >= sql.len() || sql.as_bytes()[idx] != b'(' {
            return CteExpansion {
                sql: sql.to_string(),
                had_aggregate: false,
            };
        }

        let sub = &sql[idx + 1..];
        let (rest_after_query, _) = match balanced_parens(sub) {
            Ok(value) => value,
            Err(_) => {
                return CteExpansion {
                    sql: sql.to_string(),
                    had_aggregate: false,
                };
            }
        };
        let query_len = sub.len() - rest_after_query.len();
        let query_start = idx + 1;
        let query_end = query_start + query_len;
        let query_sql = &sql[query_start..query_end];
        let expanded = expand_aggregate_with_at(query_sql);
        if expanded.had_aggregate {
            had_aggregate = true;
        }
        if expanded.expanded_sql != query_sql {
            replacements.push((query_start, query_end, expanded.expanded_sql));
        }

        idx = query_end + 1;
        idx = skip_whitespace(sql, idx);
        if idx < sql.len() && sql.as_bytes()[idx] == b',' {
            idx += 1;
            continue;
        }
        break;
    }

    if replacements.is_empty() {
        return CteExpansion {
            sql: sql.to_string(),
            had_aggregate,
        };
    }

    let mut result = sql.to_string();
    replacements.sort_by(|a, b| b.0.cmp(&a.0));
    for (start, end, replacement) in replacements {
        result.replace_range(start..end, &replacement);
    }

    CteExpansion {
        sql: result,
        had_aggregate,
    }
}

fn extract_base_relation_sql(view_query: &str) -> Option<String> {
    let query = view_query.trim().trim_end_matches(';').trim();
    if query.is_empty() {
        return None;
    }

    let has_set_op = ["UNION", "INTERSECT", "EXCEPT"]
        .iter()
        .any(|kw| find_top_level_keyword(query, kw, 0).is_some());
    if has_set_op {
        return Some(format!("SELECT * FROM ({query})"));
    }

    let select_pos = find_top_level_keyword(query, "SELECT", 0)?;
    let from_pos = find_top_level_keyword(query, "FROM", select_pos)?;
    let from_start = from_pos + 4;
    let from_end = find_first_top_level_keyword(
        query,
        from_start,
        &[
            "WHERE",
            "GROUP BY",
            "HAVING",
            "QUALIFY",
            "ORDER BY",
            "LIMIT",
            "WINDOW",
            "UNION",
            "INTERSECT",
            "EXCEPT",
        ],
    )
    .unwrap_or(query.len());

    let from_clause = query[from_start..from_end].trim();
    if from_clause.is_empty() {
        return None;
    }

    let where_clause = find_top_level_keyword(query, "WHERE", from_pos).map(|pos| {
        let where_start = pos + 5;
        let where_end = find_first_top_level_keyword(
            query,
            where_start,
            &[
                "GROUP BY",
                "HAVING",
                "QUALIFY",
                "ORDER BY",
                "LIMIT",
                "WINDOW",
                "UNION",
                "INTERSECT",
                "EXCEPT",
            ],
        )
        .unwrap_or(query.len());
        query[where_start..where_end].trim().to_string()
    });

    let cte_prefix = query[..select_pos].trim();
    let mut base_sql = String::new();
    if !cte_prefix.is_empty() {
        base_sql.push_str(cte_prefix);
        base_sql.push(' ');
    }
    base_sql.push_str("SELECT * FROM ");
    base_sql.push_str(from_clause);
    if let Some(w) = where_clause {
        if !w.is_empty() {
            base_sql.push_str(" WHERE ");
            base_sql.push_str(&w);
        }
    }

    Some(base_sql)
}

fn normalize_group_by_col(col: &str) -> String {
    let trimmed = col.trim();
    let unquoted = trimmed
        .strip_prefix('"')
        .and_then(|s| s.strip_suffix('"'))
        .unwrap_or(trimmed);
    if unquoted.contains('(') {
        let mut result = String::new();
        let mut last_space = false;
        for ch in unquoted.chars() {
            if ch.is_whitespace() {
                if !last_space {
                    result.push(' ');
                    last_space = true;
                }
            } else {
                result.push(ch.to_ascii_lowercase());
                last_space = false;
            }
        }
        result.trim().to_string()
    } else {
        let base = unquoted.split('.').next_back().unwrap_or(unquoted);
        base.to_lowercase()
    }
}

fn extract_view_group_by_cols(view_query: &str) -> Vec<String> {
    let query = view_query.trim().trim_end_matches(';').trim();
    let group_pos = match find_top_level_keyword(query, "GROUP BY", 0) {
        Some(pos) => pos,
        None => {
            if let Ok(info) = parser_ffi::parse_select(query) {
                return info
                    .items
                    .iter()
                    .filter(|item| !item.is_aggregate && !item.is_star && !item.is_measure_ref)
                    .map(|item| {
                        item.alias
                            .clone()
                            .unwrap_or_else(|| item.expression_sql.clone())
                    })
                    .collect();
            }
            return Vec::new();
        }
    };

    let start = advance_after_group_by(query, group_pos)
        .unwrap_or_else(|| group_pos + "GROUP BY".len());
    let end = find_first_top_level_keyword(
        query,
        start,
        &[
            "HAVING",
            "QUALIFY",
            "ORDER BY",
            "LIMIT",
            "WINDOW",
            "UNION",
            "INTERSECT",
            "EXCEPT",
        ],
    )
    .unwrap_or(query.len());

    let group_content = query[start..end].trim();
    if group_content.is_empty() {
        return Vec::new();
    }

    let group_upper = group_content.to_uppercase();
    if group_upper == "ALL" || group_upper.starts_with("ALL ") {
        return extract_dimension_columns_from_select(query);
    }

    let mut columns = Vec::new();
    let mut depth = 0;
    let mut current = String::new();
    for c in group_content.chars() {
        match c {
            '(' => {
                depth += 1;
                current.push(c);
            }
            ')' => {
                depth -= 1;
                current.push(c);
            }
            ',' if depth == 0 => {
                let col = current.trim();
                if !col.is_empty() && !col.chars().all(|ch| ch.is_ascii_digit()) {
                    columns.push(col.to_string());
                }
                current.clear();
            }
            _ => current.push(c),
        }
    }
    let col = current.trim();
    if !col.is_empty() && !col.chars().all(|ch| ch.is_ascii_digit()) {
        columns.push(col.to_string());
    }

    columns
}

fn find_from_clause_end(sql: &str) -> Option<usize> {
    // Preserve leading whitespace so returned indices match the original SQL.
    let query = sql.trim_end();
    let query = query.strip_suffix(';').unwrap_or(query);
    let from_pos = find_top_level_keyword(query, "FROM", 0)?;
    let start = from_pos + 4;
    Some(
        find_first_top_level_keyword(
            query,
            start,
            &[
                "WHERE",
                "GROUP BY",
                "HAVING",
                "QUALIFY",
                "ORDER BY",
                "LIMIT",
                "WINDOW",
                "UNION",
                "INTERSECT",
                "EXCEPT",
            ],
        )
        .unwrap_or(query.len()),
    )
}

fn group_by_matches_view(outer_cols: &[String], view_cols: &[String]) -> bool {
    if view_cols.is_empty() || outer_cols.is_empty() {
        return false;
    }

    let outer_set: std::collections::HashSet<String> =
        outer_cols.iter().map(|c| normalize_group_by_col(c)).collect();
    let view_set: std::collections::HashSet<String> =
        view_cols.iter().map(|c| normalize_group_by_col(c)).collect();

    !outer_set.is_empty() && outer_set == view_set
}

fn filter_group_by_cols_for_measure(
    outer_cols: &[String],
    view_cols: &[String],
    dimension_exprs: &HashMap<String, String>,
) -> Vec<String> {
    if view_cols.is_empty() {
        return outer_cols.to_vec();
    }

    let view_set: HashSet<String> = view_cols
        .iter()
        .map(|col| normalize_group_by_col(col))
        .collect();

    let mut alias_expr_norms: Vec<(String, String)> = Vec::new();
    alias_expr_norms.reserve(dimension_exprs.len());
    for (alias, expr) in dimension_exprs.iter() {
        alias_expr_norms.push((
            normalize_group_by_col(alias),
            normalize_group_by_col(expr),
        ));
    }

    outer_cols
        .iter()
        .filter(|col| {
            let normalized_outer = normalize_group_by_col(col);
            if view_set.contains(&normalized_outer) {
                return true;
            }

            if let Some(expr) = dimension_exprs.get(&normalized_outer) {
                let normalized_expr = normalize_group_by_col(expr);
                if view_set.contains(&normalized_expr) {
                    return true;
                }
            }

            alias_expr_norms.iter().any(|(alias_norm, expr_norm)| {
                expr_norm == &normalized_outer && view_set.contains(alias_norm)
            })
        })
        .cloned()
        .collect()
}

fn source_dimension_names(source_view: &str) -> HashSet<String> {
    let views = MEASURE_VIEWS.lock().unwrap();
    let Some((_, view)) = views
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case(source_view))
    else {
        return HashSet::new();
    };

    let view_query = extract_view_query(&view.base_query).unwrap_or_else(|| view.base_query.clone());
    extract_dimension_columns_from_select(&view_query)
        .into_iter()
        .map(|col| {
            let dim_name = col.split('.').next_back().unwrap_or(col.as_str()).trim();
            normalize_dimension_key(dim_name)
        })
        .collect()
}

fn can_use_view_measure_directly(resolved: &ResolvedMeasure, outer_group_by: &[String]) -> bool {
    group_by_matches_view(outer_group_by, &resolved.view_group_by_cols)
}

/// Extract WHERE clause from SQL query
pub fn extract_where_clause(sql: &str) -> Option<String> {
    let query = sql.trim().trim_end_matches(';').trim();
    let where_pos = find_top_level_keyword(query, "WHERE", 0)?;
    let start = where_pos + 5;
    let end = find_first_top_level_keyword(
        query,
        start,
        &[
            "GROUP BY",
            "HAVING",
            "QUALIFY",
            "ORDER BY",
            "LIMIT",
            "WINDOW",
            "UNION",
            "INTERSECT",
            "EXCEPT",
        ],
    )
    .unwrap_or(query.len());

    let where_content = query[start..end].trim().to_string();
    if where_content.is_empty() {
        None
    } else {
        Some(where_content)
    }
}

// =============================================================================
// Nom Parsers - Expression Helpers
// =============================================================================

/// Parse aggregation function name - accepts any identifier followed by (
/// This allows all DuckDB aggregate functions to work as measures
fn agg_function_name(input: &str) -> IResult<&str, &str> {
    // Match any identifier (alphanumeric + underscore) that's followed by (
    let (remaining, name) = nom::bytes::complete::take_while1(|c: char| c.is_alphanumeric() || c == '_')(input)?;
    // Verify it's followed by a paren (but don't consume it)
    let _ = nom::character::complete::char('(')(remaining)?;
    Ok((remaining, name))
}

/// Extract aggregation function from expression like "SUM(amount)"
pub fn extract_agg_function(expr: &str) -> String {
    agg_function_name(expr.trim())
        .map(|(_, name)| name.to_uppercase())
        .unwrap_or_else(|_| "SUM".to_string())
}

/// Extract aggregation function name from full expression
/// "SUM(amount)" -> "sum"
/// "COUNT(DISTINCT x)" -> "count"
pub fn extract_aggregation_function(expr: &str) -> Option<String> {
    let (_, (name, _)) = function_call(expr.trim()).ok()?;
    Some(name.to_lowercase())
}

/// Check if expression contains DISTINCT modifier
/// "COUNT(DISTINCT region)" -> true
fn has_distinct_modifier(expr: &str) -> bool {
    let expr_upper = expr.to_uppercase();
    // Look for DISTINCT after opening paren
    if let Some(paren_pos) = expr_upper.find('(') {
        let after_paren = &expr_upper[paren_pos + 1..];
        after_paren.trim_start().starts_with("DISTINCT")
    } else {
        false
    }
}

/// Non-decomposable aggregate functions that require recompute from base rows
const NON_DECOMPOSABLE_AGGREGATES: &[&str] = &[
    "MEDIAN",
    "PERCENTILE_CONT",
    "PERCENTILE_DISC",
    "MODE",
    "QUANTILE",
    "QUANTILE_CONT",
    "QUANTILE_DISC",
];

/// Returns true if expression uses a non-decomposable aggregate
/// (including COUNT DISTINCT and ordered-set aggregates)
fn is_non_decomposable(expr: &str) -> bool {
    if has_distinct_modifier(expr) {
        return true;
    }

    let expr_upper = expr.to_uppercase();
    NON_DECOMPOSABLE_AGGREGATES
        .iter()
        .any(|agg| expr_upper.contains(&format!("{agg}(")))
}

/// Find any aggregation function inside an expression
/// "CASE WHEN SUM(x) > 100 THEN 1 ELSE 0 END" -> Some("sum")
fn find_aggregation_in_expression(expr: &str) -> Option<String> {
    // Look for any function call pattern: identifier followed by (
    // This allows all DuckDB aggregate functions to work
    let expr_upper = expr.to_uppercase();

    // Common aggregates to check first (for performance)
    let common_aggs = [
        "SUM", "COUNT", "AVG", "MIN", "MAX", "MEDIAN", "STDDEV", "STDDEV_POP",
        "STDDEV_SAMP", "VARIANCE", "VAR_POP", "VAR_SAMP", "STRING_AGG",
        "ARRAY_AGG", "LIST", "FIRST", "LAST", "MODE", "QUANTILE",
    ];

    for agg in common_aggs {
        if expr_upper.contains(&format!("{agg}(")) {
            return Some(agg.to_lowercase());
        }
    }

    // Fallback: look for any identifier followed by ( that could be an aggregate
    // This catches custom aggregates
    let chars: Vec<char> = expr.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        // Look for start of identifier
        if chars[i].is_alphabetic() || chars[i] == '_' {
            let start = i;
            while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            // Check if followed by (
            if i < chars.len() && chars[i] == '(' {
                let name: String = chars[start..i].iter().collect();
                // Skip known non-aggregates
                let name_upper = name.to_uppercase();
                if !["CASE", "WHEN", "THEN", "ELSE", "END", "AND", "OR", "NOT", "IN", "IS", "NULL", "TRUE", "FALSE", "LIKE", "BETWEEN", "CAST", "COALESCE", "NULLIF", "IF", "IIF"].contains(&name_upper.as_str()) {
                    return Some(name.to_lowercase());
                }
            }
        }
        i += 1;
    }

    None
}

/// Check if an expression contains COUNT(DISTINCT ...) which is non-decomposable
/// "COUNT(DISTINCT user_id)" -> true
/// "COUNT(user_id)" -> false
/// "SUM(amount)" -> false
pub fn is_count_distinct(expr: &str) -> bool {
    let expr_upper = expr.to_uppercase();
    // Match COUNT followed by optional whitespace, (, optional whitespace, DISTINCT
    // This handles: COUNT(DISTINCT x), COUNT( DISTINCT x), COUNT (DISTINCT x)
    let patterns = ["COUNT(DISTINCT", "COUNT( DISTINCT", "COUNT (DISTINCT"];
    patterns.iter().any(|p| expr_upper.contains(p))
}

/// Expand derived measure expression by replacing measure references with their aggregations
/// "revenue - cost" with measures [revenue=SUM(amount), cost=SUM(expense)]
/// -> "SUM(revenue) - SUM(cost)"
fn expand_derived_measure_expr(expr: &str, measure_view: &MeasureView) -> String {
    let mut result = String::new();
    let mut chars = expr.chars().peekable();

    while let Some(c) = chars.next() {
        if c == '\'' {
            result.push(c);
            while let Some(next) = chars.next() {
                result.push(next);
                if next == '\'' {
                    if chars.peek() == Some(&'\'') {
                        result.push(chars.next().unwrap());
                    } else {
                        break;
                    }
                }
            }
            continue;
        }
        if c == '"' {
            result.push(c);
            while let Some(next) = chars.next() {
                result.push(next);
                if next == '"' {
                    if chars.peek() == Some(&'"') {
                        result.push(chars.next().unwrap());
                    } else {
                        break;
                    }
                }
            }
            continue;
        }
        if c == '`' {
            result.push(c);
            while let Some(next) = chars.next() {
                result.push(next);
                if next == '`' {
                    break;
                }
            }
            continue;
        }
        if c == '[' {
            result.push(c);
            while let Some(next) = chars.next() {
                result.push(next);
                if next == ']' {
                    break;
                }
            }
            continue;
        }
        if c.is_alphabetic() || c == '_' {
            // Collect identifier
            let mut ident = String::from(c);
            while let Some(&next) = chars.peek() {
                if next.is_alphanumeric() || next == '_' {
                    ident.push(chars.next().unwrap());
                } else {
                    break;
                }
            }

            // Check if this identifier is a measure name
            if let Some(m) = measure_view
                .measures
                .iter()
                .find(|m| m.column_name.eq_ignore_ascii_case(&ident))
            {
                result.push('(');
                result.push_str(&m.expression);
                result.push(')');
            } else {
                // Not a measure, keep as-is
                result.push_str(&ident);
            }
        } else {
            result.push(c);
        }
    }

    result
}

/// Qualify dimension reference in expression for correlated subquery
/// "year - 1" with table "sales" and dim "year" -> "sales.year - 1"
pub fn qualify_outer_reference(expr: &str, table_name: &str, dim: &str) -> String {
    // Parse expression into tokens and replace matching identifiers
    let mut result = String::new();
    let mut chars = expr.chars().peekable();

    while let Some(c) = chars.next() {
        if c.is_alphabetic() || c == '_' {
            // Collect identifier
            let mut ident = String::from(c);
            while let Some(&next) = chars.peek() {
                if next.is_alphanumeric() || next == '_' {
                    ident.push(chars.next().unwrap());
                } else {
                    break;
                }
            }

            // Check if this identifier matches the dimension
            if ident == dim {
                result.push_str(table_name);
                result.push('.');
            }
            result.push_str(&ident);
        } else {
            result.push(c);
        }
    }

    result
}

fn expr_mentions_identifier(expr: &str, ident: &str) -> bool {
    let mut chars = expr.chars().peekable();

    while let Some(c) = chars.next() {
        if c == '\'' {
            // Skip SQL single-quoted string literal content, including escaped ''.
            while let Some(next) = chars.next() {
                if next == '\'' {
                    if chars.peek() == Some(&'\'') {
                        chars.next();
                    } else {
                        break;
                    }
                }
            }
            continue;
        }

        if c.is_alphabetic() || c == '_' {
            let mut token = String::from(c);
            while let Some(&next) = chars.peek() {
                if next.is_alphanumeric() || next == '_' {
                    token.push(chars.next().unwrap());
                } else {
                    break;
                }
            }

            if token.eq_ignore_ascii_case(ident) {
                return true;
            }
        }
    }

    false
}

fn dimension_in_group_by(
    dim: &str,
    group_by_cols: &[String],
    default_qualifier: Option<&str>,
) -> bool {
    let dim_trim = dim.trim();
    let dim_name = dim_trim.split('.').next_back().unwrap_or(dim_trim).trim();
    let explicit_dim_qualifier = dim_trim
        .rsplit_once('.')
        .map(|(qualifier, _)| qualifier.trim());
    let expected_qualifier = explicit_dim_qualifier.or(default_qualifier);
    let dim_lower = dim_trim.to_lowercase();

    if dim_trim.contains('(') {
        return group_by_cols
            .iter()
            .any(|col| col.to_lowercase() == dim_lower);
    }

    group_by_cols.iter().any(|col| {
        let col_trim = col.trim();
        let col_name = col_trim.split('.').next_back().unwrap_or(col_trim).trim();

        if !col_name.eq_ignore_ascii_case(dim_name) {
            return false;
        }

        // When we know the expected outer qualifier, GROUP BY qualifiers must match it.
        // Unqualified GROUP BY columns still count.
        match (expected_qualifier, col_trim.rsplit_once('.')) {
            (Some(expected), Some((col_qualifier, _))) => {
                col_qualifier.trim().eq_ignore_ascii_case(expected)
            }
            _ => true,
        }
    })
}

fn expr_mentions_identifier_outside_current(expr: &str, ident: &str) -> bool {
    let mut chars = expr.chars().peekable();
    let mut pending_current = false;

    while let Some(c) = chars.next() {
        if c == '\'' {
            while let Some(next) = chars.next() {
                if next == '\'' {
                    if chars.peek() == Some(&'\'') {
                        chars.next();
                    } else {
                        break;
                    }
                }
            }
            continue;
        }

        if c.is_alphabetic() || c == '_' {
            let mut token = String::from(c);
            while let Some(&next) = chars.peek() {
                if next.is_alphanumeric() || next == '_' {
                    token.push(chars.next().unwrap());
                } else {
                    break;
                }
            }

            if token.eq_ignore_ascii_case("CURRENT") {
                pending_current = true;
                continue;
            }

            if token.eq_ignore_ascii_case(ident) {
                if pending_current {
                    pending_current = false;
                } else {
                    return true;
                }
            } else {
                pending_current = false;
            }
        }
    }

    false
}

fn where_has_simple_equality_constraint(
    where_clause: &str,
    dim_name: &str,
    default_qualifier: Option<&str>,
) -> bool {
    // Conservative: if OR is present we don't claim single-valued context.
    if where_clause.to_uppercase().contains(" OR ") {
        return false;
    }

    let expected_qualifier = default_qualifier.map(normalize_identifier_name);
    let bytes = where_clause.as_bytes();
    let mut i = 0usize;

    while i < bytes.len() {
        if bytes[i] == b'=' {
            if (i > 0 && matches!(bytes[i - 1], b'<' | b'>' | b'!' | b'='))
                || (i + 1 < bytes.len() && bytes[i + 1] == b'=')
            {
                i += 1;
                continue;
            }

            let mut end = i;
            while end > 0 && bytes[end - 1].is_ascii_whitespace() {
                end -= 1;
            }
            let mut start = end;
            while start > 0 && is_measure_ref_char(bytes[start - 1]) {
                start -= 1;
            }

            if start < end {
                let left = where_clause[start..end].trim();
                if let Some((qualifier, name)) = parse_simple_measure_ref(left) {
                    if name.eq_ignore_ascii_case(dim_name) {
                        let qualifier_ok = match (&expected_qualifier, qualifier) {
                            (Some(expected), Some(found)) => found.eq_ignore_ascii_case(expected),
                            _ => true,
                        };
                        if qualifier_ok {
                            return true;
                        }
                    }
                }
            }
        }
        i += 1;
    }

    false
}

fn current_dimension_is_single_valued(
    dim: &str,
    group_by_cols: &[String],
    outer_where: Option<&str>,
    default_qualifier: Option<&str>,
) -> bool {
    if dimension_in_group_by(dim, group_by_cols, default_qualifier) {
        return true;
    }

    let dim_name = dim.split('.').next_back().unwrap_or(dim).trim();
    outer_where
        .map(|w| where_has_simple_equality_constraint(w, dim_name, default_qualifier))
        .unwrap_or(false)
}

fn resolve_current_in_expr(
    expr: &str,
    group_by_cols: &[String],
    outer_where: Option<&str>,
    default_qualifier: Option<&str>,
) -> String {
    let bytes = expr.as_bytes();
    let mut out = String::new();
    let mut i = 0usize;

    while i < bytes.len() {
        match bytes[i] {
            b'\'' => {
                out.push('\'');
                i += 1;
                while i < bytes.len() {
                    out.push(bytes[i] as char);
                    if bytes[i] == b'\'' {
                        if i + 1 < bytes.len() && bytes[i + 1] == b'\'' {
                            i += 1;
                            out.push(bytes[i] as char);
                        } else {
                            i += 1;
                            break;
                        }
                    }
                    i += 1;
                }
            }
            c if (c as char).is_alphabetic() || c == b'_' => {
                let token_start = i;
                i += 1;
                while i < bytes.len() && ((bytes[i] as char).is_alphanumeric() || bytes[i] == b'_') {
                    i += 1;
                }
                let token = &expr[token_start..i];

                if token.eq_ignore_ascii_case("CURRENT") {
                    let mut j = i;
                    while j < bytes.len() && bytes[j].is_ascii_whitespace() {
                        j += 1;
                    }

                    if j < bytes.len()
                        && (((bytes[j] as char).is_alphabetic()) || bytes[j] == b'_')
                    {
                        let dim_start = j;
                        j += 1;
                        while j < bytes.len()
                            && ((bytes[j] as char).is_alphanumeric() || bytes[j] == b'_')
                        {
                            j += 1;
                        }
                        while j < bytes.len() && bytes[j] == b'.' {
                            let mut k = j + 1;
                            if k >= bytes.len()
                                || !(((bytes[k] as char).is_alphabetic()) || bytes[k] == b'_')
                            {
                                break;
                            }
                            k += 1;
                            while k < bytes.len()
                                && ((bytes[k] as char).is_alphanumeric() || bytes[k] == b'_')
                            {
                                k += 1;
                            }
                            j = k;
                        }

                        let dim = expr[dim_start..j].trim();
                        if current_dimension_is_single_valued(
                            dim,
                            group_by_cols,
                            outer_where,
                            default_qualifier,
                        ) {
                            out.push_str(dim);
                        } else {
                            out.push_str("NULL");
                        }
                        i = j;
                        continue;
                    }
                }

                out.push_str(token);
            }
            _ => {
                out.push(bytes[i] as char);
                i += 1;
            }
        }
    }

    out
}

/// Qualify column references in a WHERE clause for use inside _inner subquery
/// "region = 'US'" -> "_inner.region = 'US'"
/// "year > 2020 AND region = 'US'" -> "_inner.year > 2020 AND _inner.region = 'US'"
fn qualify_where_for_inner(where_clause: &str) -> String {
    if let Ok(result) = parser_ffi::qualify_expression(where_clause, "_inner") {
        return result;
    }
    qualify_where_for_inner_fallback(where_clause)
}

fn qualify_where_for_inner_fallback(where_clause: &str) -> String {
    let mut result = String::new();
    let mut chars = where_clause.chars().peekable();
    let mut previous_was_dot = false;
    let mut previous_was_as = false;

    while let Some(c) = chars.next() {
        if c.is_alphabetic() || c == '_' {
            // Collect identifier
            let mut ident = String::from(c);
            while let Some(&next) = chars.peek() {
                if next.is_alphanumeric() || next == '_' {
                    ident.push(chars.next().unwrap());
                } else {
                    break;
                }
            }

            // Check if followed by a dot (already qualified) or paren (function call)
            let already_qualified = previous_was_dot || chars.peek() == Some(&'.');
            let is_function = chars.peek() == Some(&'(');

            // SQL keywords that should not be prefixed
            let keywords = [
                "AND", "OR", "NOT", "IN", "IS", "AS", "NULL", "TRUE", "FALSE", "LIKE", "BETWEEN",
                "EXISTS", "CASE", "WHEN", "THEN", "ELSE", "END",
            ];
            let is_keyword = keywords.iter().any(|kw| kw.eq_ignore_ascii_case(&ident));
            let is_type_context = previous_was_as;
            let is_typed_literal = is_typed_literal_keyword(&ident, &chars);

            if !already_qualified && !is_keyword && !is_function && !is_typed_literal && !is_type_context {
                result.push_str("_inner.");
            }
            result.push_str(&ident);
            previous_was_dot = false;
            previous_was_as = ident.eq_ignore_ascii_case("AS");
        } else if c == '\'' {
            // String literal - copy as-is until closing quote
            result.push(c);
            previous_was_dot = false;
            while let Some(next) = chars.next() {
                result.push(next);
                if next == '\'' {
                    // Check for escaped quote ''
                    if chars.peek() == Some(&'\'') {
                        result.push(chars.next().unwrap());
                    } else {
                        break;
                    }
                }
            }
        } else {
            result.push(c);
            previous_was_dot = c == '.';
        }
    }

    result
}

fn qualify_where_for_outer(where_clause: &str, outer_alias: &str) -> String {
    if let Ok(result) = parser_ffi::qualify_expression(where_clause, outer_alias) {
        return result;
    }
    qualify_where_for_outer_fallback(where_clause, outer_alias)
}

fn qualify_where_for_outer_fallback(where_clause: &str, outer_alias: &str) -> String {
    let mut result = String::new();
    let mut chars = where_clause.chars().peekable();
    let mut previous_was_dot = false;
    let mut previous_was_as = false;

    while let Some(c) = chars.next() {
        if c.is_alphabetic() || c == '_' {
            let mut ident = String::from(c);
            while let Some(&next) = chars.peek() {
                if next.is_alphanumeric() || next == '_' {
                    ident.push(chars.next().unwrap());
                } else {
                    break;
                }
            }

            let already_qualified = previous_was_dot || chars.peek() == Some(&'.');
            let is_function = chars.peek() == Some(&'(');

            let keywords = [
                "AND", "OR", "NOT", "IN", "IS", "AS", "NULL", "TRUE", "FALSE", "LIKE", "BETWEEN",
                "EXISTS", "CASE", "WHEN", "THEN", "ELSE", "END",
            ];
            let is_keyword = keywords.iter().any(|kw| kw.eq_ignore_ascii_case(&ident));
            let is_type_context = previous_was_as;
            let is_typed_literal = is_typed_literal_keyword(&ident, &chars);

            if !already_qualified && !is_keyword && !is_function && !is_typed_literal && !is_type_context {
                result.push_str(outer_alias);
                result.push('.');
            }
            result.push_str(&ident);
            previous_was_dot = false;
            previous_was_as = ident.eq_ignore_ascii_case("AS");
        } else if c == '\'' {
            result.push(c);
            previous_was_dot = false;
            while let Some(next) = chars.next() {
                result.push(next);
                if next == '\'' {
                    if chars.peek() == Some(&'\'') {
                        result.push(chars.next().unwrap());
                    } else {
                        break;
                    }
                }
            }
        } else {
            result.push(c);
            previous_was_dot = c == '.';
        }
    }

    result
}

fn strip_at_where_qualifiers(condition: &str) -> String {
    let mut result = String::new();
    let mut chars = condition.chars().peekable();

    while let Some(c) = chars.next() {
        if c.is_alphabetic() || c == '_' {
            let mut ident = String::from(c);
            while let Some(&next) = chars.peek() {
                if next.is_alphanumeric() || next == '_' {
                    ident.push(chars.next().unwrap());
                } else {
                    break;
                }
            }

            let mut last_ident = ident;
            while chars.peek() == Some(&'.') {
                chars.next(); // consume '.'
                if let Some(&next) = chars.peek() {
                    if next.is_alphabetic() || next == '_' {
                        let mut next_ident = String::new();
                        while let Some(&next_ch) = chars.peek() {
                            if next_ch.is_alphanumeric() || next_ch == '_' {
                                next_ident.push(chars.next().unwrap());
                            } else {
                                break;
                            }
                        }
                        last_ident = next_ident;
                        continue;
                    } else {
                        result.push_str(&last_ident);
                        result.push('.');
                        break;
                    }
                } else {
                    result.push_str(&last_ident);
                    result.push('.');
                    break;
                }
            }

            result.push_str(&last_ident);
        } else if c == '\'' {
            result.push(c);
            while let Some(next) = chars.next() {
                result.push(next);
                if next == '\'' {
                    if chars.peek() == Some(&'\'') {
                        result.push(chars.next().unwrap());
                    } else {
                        break;
                    }
                }
            }
        } else {
            result.push(c);
        }
    }

    result
}

fn rewrite_percentile_within_group(sql: &str) -> String {
    let upper = sql.to_uppercase();
    let upper_bytes = upper.as_bytes();
    let bytes = sql.as_bytes();
    let mut out = String::new();
    let mut i = 0;

    while i < bytes.len() {
        let (func_name, quantile_name) = if upper_bytes[i..].starts_with(b"PERCENTILE_CONT") {
            ("PERCENTILE_CONT", "QUANTILE_CONT")
        } else if upper_bytes[i..].starts_with(b"PERCENTILE_DISC") {
            ("PERCENTILE_DISC", "QUANTILE_DISC")
        } else {
            out.push(bytes[i] as char);
            i += 1;
            continue;
        };

        let name_len = func_name.len();
        let mut j = i + name_len;
        while j < bytes.len() && bytes[j].is_ascii_whitespace() {
            j += 1;
        }
        if j >= bytes.len() || bytes[j] != b'(' {
            out.push(bytes[i] as char);
            i += 1;
            continue;
        }

        let args_start = j + 1;
        let (after_args, args) = match balanced_parens(&sql[args_start..]) {
            Ok(res) => res,
            Err(_) => {
                out.push(bytes[i] as char);
                i += 1;
                continue;
            }
        };
        let args_len = args.len();
        let args_end = args_start + args_len;
        if args_end >= sql.len() || !after_args.starts_with(')') {
            out.push(bytes[i] as char);
            i += 1;
            continue;
        }

        let mut k = args_end + 1;
        while k < bytes.len() && bytes[k].is_ascii_whitespace() {
            k += 1;
        }
        if k >= bytes.len() || !upper_bytes[k..].starts_with(b"WITHIN") {
            out.push(bytes[i] as char);
            i += 1;
            continue;
        }
        k += "WITHIN".len();
        while k < bytes.len() && bytes[k].is_ascii_whitespace() {
            k += 1;
        }
        if k >= bytes.len() || !upper_bytes[k..].starts_with(b"GROUP") {
            out.push(bytes[i] as char);
            i += 1;
            continue;
        }
        k += "GROUP".len();
        while k < bytes.len() && bytes[k].is_ascii_whitespace() {
            k += 1;
        }
        if k >= bytes.len() || bytes[k] != b'(' {
            out.push(bytes[i] as char);
            i += 1;
            continue;
        }

        let inner_start = k + 1;
        let (after_inner, inner) = match balanced_parens(&sql[inner_start..]) {
            Ok(res) => res,
            Err(_) => {
                out.push(bytes[i] as char);
                i += 1;
                continue;
            }
        };
        let inner_len = inner.len();
        let inner_end = inner_start + inner_len;
        if inner_end >= sql.len() || !after_inner.starts_with(')') {
            out.push(bytes[i] as char);
            i += 1;
            continue;
        }

        let inner_trim = inner.trim();
        let inner_upper = inner_trim.to_uppercase();
        if !inner_upper.starts_with("ORDER BY") {
            out.push(bytes[i] as char);
            i += 1;
            continue;
        }
        let order_expr = inner_trim["ORDER BY".len()..].trim();
        if order_expr.is_empty() {
            out.push(bytes[i] as char);
            i += 1;
            continue;
        }

        out.push_str(&format!(
            "{quantile_name}({order_expr}, {})",
            args.trim()
        ));
        i = inner_end + 1;
    }

    out
}

fn qualify_where_for_inner_with_dimensions(
    where_clause: &str,
    dimension_exprs: &HashMap<String, String>,
) -> String {
    let mut result = String::new();
    let mut chars = where_clause.chars().peekable();
    let mut previous_was_dot = false;
    let mut previous_was_as = false;

    while let Some(c) = chars.next() {
        if c.is_alphabetic() || c == '_' {
            let mut ident = String::from(c);
            while let Some(&next) = chars.peek() {
                if next.is_alphanumeric() || next == '_' {
                    ident.push(chars.next().unwrap());
                } else {
                    break;
                }
            }

            let already_qualified = previous_was_dot || chars.peek() == Some(&'.');
            let is_function = chars.peek() == Some(&'(');

            let keywords = [
                "AND", "OR", "NOT", "IN", "IS", "AS", "NULL", "TRUE", "FALSE", "LIKE", "BETWEEN",
                "EXISTS", "CASE", "WHEN", "THEN", "ELSE", "END",
            ];
            let is_keyword = keywords.iter().any(|kw| kw.eq_ignore_ascii_case(&ident));
            let is_type_context = previous_was_as;
            let is_typed_literal = is_typed_literal_keyword(&ident, &chars);

            if !already_qualified && !is_keyword && !is_function && !is_typed_literal && !is_type_context {
                let key = normalize_dimension_key(&ident);
                if let Some(expr) = dimension_exprs.get(&key) {
                    let inner_expr = qualify_where_for_inner(expr);
                    result.push('(');
                    result.push_str(&inner_expr);
                    result.push(')');
                    continue;
                }
                result.push_str("_inner.");
            }
            result.push_str(&ident);
            previous_was_dot = false;
            previous_was_as = ident.eq_ignore_ascii_case("AS");
        } else if c == '\'' {
            result.push(c);
            previous_was_dot = false;
            while let Some(next) = chars.next() {
                result.push(next);
                if next == '\'' {
                    if chars.peek() == Some(&'\'') {
                        result.push(chars.next().unwrap());
                    } else {
                        break;
                    }
                }
            }
        } else {
            result.push(c);
            previous_was_dot = c == '.';
        }
    }

    result
}

fn is_typed_literal_keyword(
    ident: &str,
    chars: &std::iter::Peekable<std::str::Chars<'_>>,
) -> bool {
    let is_keyword = ident.eq_ignore_ascii_case("DATE")
        || ident.eq_ignore_ascii_case("TIME")
        || ident.eq_ignore_ascii_case("TIMESTAMP")
        || ident.eq_ignore_ascii_case("TIMESTAMPTZ")
        || ident.eq_ignore_ascii_case("INTERVAL");
    if !is_keyword {
        return false;
    }

    let mut lookahead = chars.clone();
    while let Some(&next) = lookahead.peek() {
        if next.is_whitespace() {
            lookahead.next();
            continue;
        }
        break;
    }

    matches!(lookahead.peek(), Some(&'\''))
}

// =============================================================================
// Core Functions - Process CREATE VIEW
// =============================================================================

/// Process CREATE VIEW statement, extracting AS MEASURE definitions
pub fn process_create_view(sql: &str) -> CreateViewResult {
    if !has_as_measure(sql) {
        return CreateViewResult {
            is_measure_view: false,
            view_name: None,
            clean_sql: sql.to_string(),
            measures: vec![],
            error: None,
        };
    }

    let result = extract_measures_from_sql(sql);

    match result {
        Ok((clean_sql, measures, view_name, base_table)) => {
            if !measures.is_empty() {
                if let Some(ref vn) = view_name {
                    let view_query =
                        extract_view_query(&clean_sql).unwrap_or_else(|| clean_sql.clone());
                    let base_relation_sql = extract_base_relation_sql(&view_query);
                    let dimension_exprs = extract_dimension_exprs_from_query(&view_query);
                    let mut group_by_cols = extract_view_group_by_cols(&view_query);
                    let measure_col_names: HashSet<String> = measures
                        .iter()
                        .map(|m| normalize_group_by_col(&m.column_name))
                        .collect();
                    group_by_cols.retain(|col| !measure_col_names.contains(&normalize_group_by_col(col)));
                    let measure_view = MeasureView {
                        view_name: vn.clone(),
                        measures: measures.clone(),
                        base_query: clean_sql.clone(),
                        base_table,
                        base_relation_sql,
                        dimension_exprs,
                        group_by_cols,
                    };

                    let mut views = MEASURE_VIEWS.lock().unwrap();
                    views.insert(vn.clone(), measure_view);
                }
            }

            CreateViewResult {
                is_measure_view: !measures.is_empty(),
                view_name,
                clean_sql,
                measures,
                error: None,
            }
        }
        Err(e) => CreateViewResult {
            is_measure_view: false,
            view_name: None,
            clean_sql: sql.to_string(),
            measures: vec![],
            error: Some(e.to_string()),
        },
    }
}

/// Extract measures from SQL using nom-based parsing
/// Returns (clean_sql, measures, view_name, base_table)
fn extract_measures_from_sql(
    sql: &str,
) -> Result<(String, Vec<ViewMeasure>, Option<String>, Option<String>)> {
    let view_name = extract_view_name(sql);
    let base_table = extract_table_name_from_sql(sql);
    let sql_upper = sql.to_uppercase();

    // First pass: collect all measures with positions
    struct MeasureInfo {
        name: String,
        expression: String,
        expr_start: usize,
        name_end: usize,
    }
    let mut measure_infos: Vec<MeasureInfo> = Vec::new();

    let mut search_pos = 0;
    while let Some(offset) = sql_upper[search_pos..].find(" AS MEASURE ") {
        let pattern_start = search_pos + offset;
        let after_measure = pattern_start + " AS MEASURE ".len();

        if let Ok((rest, name)) = identifier(&sql[after_measure..]) {
            let name_end = after_measure + (sql[after_measure..].len() - rest.len());
            let expr_start = find_expression_start(sql, pattern_start);
            let expression = sql[expr_start..pattern_start].trim().to_string();

            measure_infos.push(MeasureInfo {
                name: name.to_string(),
                expression,
                expr_start,
                name_end,
            });

            search_pos = name_end;
        } else {
            search_pos = pattern_start + 1;
        }
    }

    // Replace measured expressions:
    // - decomposable measures become NULL placeholders (virtual columns)
    // - non-decomposable measures keep their aggregate expression for direct querying
    let mut replacements: Vec<(usize, usize, String)> = Vec::new();
    let mut has_materialized_non_decomposable = false;

    for info in &measure_infos {
        let is_non_decomp = is_non_decomposable(&info.expression);
        if is_non_decomp {
            has_materialized_non_decomposable = true;
        }
        replacements.push((
            info.expr_start,
            info.name_end,
            if is_non_decomp {
                format!("{} AS {}", info.expression.trim(), info.name)
            } else {
                format!("NULL AS {}", info.name)
            },
        ));
    }

    // Build measures list
    let measures: Vec<ViewMeasure> = measure_infos
        .into_iter()
        .map(|m| ViewMeasure {
            column_name: m.name,
            is_decomposable: !is_non_decomposable(&m.expression),
            expression: m.expression,
        })
        .collect();

    // Apply replacements in reverse order
    let mut clean_sql = sql.to_string();
    replacements.sort_by(|a, b| b.0.cmp(&a.0));
    for (start, end, replacement) in replacements {
        clean_sql = format!(
            "{}{}{}",
            &clean_sql[..start],
            replacement,
            &clean_sql[end..]
        );
    }

    clean_sql = rewrite_percentile_within_group(&clean_sql);

    // Non-decomposable measures (e.g., COUNT DISTINCT, MEDIAN) kept as aggregates
    // require grouping to form a valid view if dimensions are projected.
    if has_materialized_non_decomposable && !has_group_by_anywhere(&clean_sql) {
        let upper = clean_sql.to_uppercase();
        let insert_pos = ["ORDER BY", "LIMIT", "HAVING", ";"]
            .iter()
            .filter_map(|kw| upper.find(kw))
            .min()
            .unwrap_or(clean_sql.len());
        clean_sql = format!(
            "{} GROUP BY ALL{}",
            clean_sql[..insert_pos].trim_end(),
            if insert_pos < clean_sql.len() {
                format!(" {}", clean_sql[insert_pos..].trim_start())
            } else {
                String::new()
            }
        );
    }

    Ok((clean_sql, measures, view_name, base_table))
}

/// Find the start of an expression before AS MEASURE
fn find_expression_start(sql: &str, end: usize) -> usize {
    let chars: Vec<char> = sql[..end].chars().collect();
    let mut pos = chars.len();
    let mut paren_depth = 0;

    // Skip trailing whitespace
    while pos > 0 && chars[pos - 1].is_whitespace() {
        pos -= 1;
    }

    // Walk backward through the expression
    while pos > 0 {
        let c = chars[pos - 1];

        match c {
            ')' => {
                paren_depth += 1;
                pos -= 1;
            }
            '(' => {
                if paren_depth > 0 {
                    paren_depth -= 1;
                    pos -= 1;
                } else {
                    break;
                }
            }
            ',' if paren_depth == 0 => break,
            _ if c.is_whitespace() && paren_depth == 0 => {
                // Check if previous word is a SQL keyword
                let remaining = &sql[..pos - 1];
                let trimmed = remaining.trim_end();
                if trimmed.is_empty() {
                    break;
                }
                let last_word = trimmed.split_whitespace().last().unwrap_or("");
                let last_upper = last_word.to_uppercase();
                if matches!(
                    last_upper.as_str(),
                    "SELECT" | "FROM" | "WHERE" | "GROUP" | "ORDER" | "HAVING"
                ) {
                    break;
                }
                pos -= 1;
            }
            _ => pos -= 1,
        }
    }

    // Find actual start by looking for comma
    let byte_pos: usize = chars[..pos].iter().collect::<String>().len();
    sql[..byte_pos].rfind(',').map(|p| p + 1).unwrap_or(0)
}

// =============================================================================
// Core Functions - AGGREGATE Expansion
// =============================================================================

/// Expand AGGREGATE() function calls in a SELECT statement
/// Uses position-based replacement with the C++ FFI parser
pub fn expand_aggregate(sql: &str) -> AggregateExpandResult {
    let cte_expansion = expand_cte_queries(sql);
    let sql = cte_expansion.sql;
    let mut had_aggregate = cte_expansion.had_aggregate;

    if !has_aggregate_function(&sql) {
        return AggregateExpandResult {
            had_aggregate,
            expanded_sql: sql,
            error: None,
        };
    }
    had_aggregate = true;

    // Parse the SQL to get table info and GROUP BY columns
    let select_info = match parser_ffi::parse_select(&sql) {
        Ok(info) => info,
        Err(e) => {
            return AggregateExpandResult {
                had_aggregate,
                expanded_sql: sql,
                error: Some(format!("SQL parse error: {e}")),
            };
        }
    };

    // Get the primary table name
    let table_name = select_info.primary_table.clone().unwrap_or_default();

    // Get measure view if this table has one
    let views = MEASURE_VIEWS.lock().unwrap();
    let measure_view = views.get(&table_name);

    // Extract all AGGREGATE() calls (without AT modifiers)
    let mut aggregate_calls = extract_all_aggregate_calls(&sql);

    if aggregate_calls.is_empty() {
        return AggregateExpandResult {
            had_aggregate: false,
            expanded_sql: sql.to_string(),
            error: None,
        };
    }

    // Sort by position descending for safe replacement
    aggregate_calls.sort_by(|a, b| b.1.cmp(&a.1));

    // Non-decomposable measures require the recompute path
    if let Some(mv) = measure_view {
        let uses_non_decomposable = aggregate_calls.iter().any(|(measure_name, _, _)| {
            mv.measures
                .iter()
                .find(|m| m.column_name.eq_ignore_ascii_case(measure_name))
                .map(|m| is_non_decomposable(&m.expression))
                .unwrap_or(false)
        });
        if uses_non_decomposable {
            return expand_aggregate_with_at(&sql);
        }
    }

    // Build replacements
    let mut result_sql = sql;

    for (measure_name, start, end) in aggregate_calls {
        // Look up measure definition
        let expanded = if let Some(mv) = measure_view {
            if let Some(m) = mv
                .measures
                .iter()
                .find(|m| m.column_name.eq_ignore_ascii_case(&measure_name))
            {
                // Get aggregation function from measure expression
                if let Some(agg_fn) = extract_aggregation_function(&m.expression) {
                    format!("{agg_fn}({measure_name})")
                } else if let Some(agg_fn) = find_aggregation_in_expression(&m.expression) {
                    format!("{agg_fn}({measure_name})")
                } else {
                    // Check if derived measure
                    let expanded_expr = expand_derived_measure_expr(&m.expression, mv);
                    if expanded_expr != m.expression {
                        expanded_expr
                    } else {
                        format!("SUM({measure_name})")
                    }
                }
            } else {
                format!("SUM({measure_name})")
            }
        } else {
            format!("SUM({measure_name})")
        };

        result_sql = format!("{}{}{}", &result_sql[..start], expanded, &result_sql[end..]);
    }

    // Check if we need to add GROUP BY
    let result_upper = result_sql.to_uppercase();
    if !has_group_by_anywhere(&result_sql) {
        // Extract dimension columns (non-aggregate items)
        let dim_cols = extract_dimension_columns_from_select_info(&select_info);
        if !dim_cols.is_empty() {
            // Find insertion point
            let insert_pos = ["ORDER BY", "LIMIT", "HAVING", ";"]
                .iter()
                .filter_map(|kw| result_upper.find(kw))
                .min()
                .unwrap_or(result_sql.len());

            result_sql = format!(
                "{} GROUP BY {}{}",
                result_sql[..insert_pos].trim_end(),
                dim_cols.join(", "),
                if insert_pos < result_sql.len() {
                    format!(" {}", result_sql[insert_pos..].trim_start())
                } else {
                    String::new()
                }
            );
        }
    }

    AggregateExpandResult {
        had_aggregate,
        expanded_sql: result_sql,
        error: None,
    }
}

/// Extract dimension (non-aggregate) columns from SelectInfo
fn extract_dimension_columns_from_select_info(info: &SelectInfo) -> Vec<String> {
    info.items
        .iter()
        .filter(|item| !item.is_aggregate && !item.is_star && !item.is_measure_ref)
        .filter(|item| !is_literal_constant(&item.expression_sql))
        .map(|item| {
            // Use alias if present, otherwise expression
            item.alias
                .clone()
                .unwrap_or_else(|| item.expression_sql.clone())
        })
        .collect()
}

fn normalize_dimension_key(ident: &str) -> String {
    let trimmed = ident.trim();
    let unquoted = trimmed
        .strip_prefix('"')
        .and_then(|s| s.strip_suffix('"'))
        .unwrap_or(trimmed);
    unquoted.to_lowercase()
}

fn extract_dimension_exprs_from_query(sql: &str) -> HashMap<String, String> {
    let mut exprs = HashMap::new();
    let info = std::panic::catch_unwind(|| parser_ffi::parse_select(sql))
        .ok()
        .and_then(|result| result.ok());
    if let Some(info) = info {
        for item in info
            .items
            .iter()
            .filter(|item| !item.is_aggregate && !item.is_star && !item.is_measure_ref)
        {
            if let Some(alias) = &item.alias {
                exprs.insert(normalize_dimension_key(alias), item.expression_sql.clone());
            }
        }
    }
    exprs
}

// =============================================================================
// FROM clause extraction (for JOIN support)
// =============================================================================

/// Extract all tables from a SELECT's FROM clause using the FFI parser
/// Returns a FromClauseInfo with tables from the parsed SQL
#[allow(dead_code)] // Used in tests (marked as ignore, require C++ library)
fn extract_from_clause_info_ffi(sql: &str) -> FromClauseInfo {
    let mut info = FromClauseInfo::default();

    if let Ok(select_info) = parser_ffi::parse_select(sql) {
        for (i, table_ref) in select_info.tables.iter().enumerate() {
            let effective_name = table_ref
                .alias
                .clone()
                .unwrap_or_else(|| table_ref.table_name.clone());
            let has_alias = table_ref.alias.is_some();

            let table_info = TableInfo {
                name: table_ref.table_name.clone(),
                effective_name: effective_name.clone(),
                has_alias,
            };

            if i == 0 && info.primary_table.is_none() {
                info.primary_table = Some(table_info.clone());
            }
            info.tables.insert(effective_name, table_info);
        }
    }

    info
}

/// Information about a resolved measure
#[derive(Debug, Clone)]
pub struct ResolvedMeasure {
    /// The aggregation function (SUM, COUNT, etc.)
    pub agg_fn: String,
    /// The source view name
    pub source_view: String,
    /// For derived measures: the expanded expression
    pub derived_expr: Option<String>,
    /// Whether this measure can be re-aggregated
    pub is_decomposable: bool,
    /// Base table for non-decomposable measures (for correlated subquery)
    pub base_table: Option<String>,
    /// Base relation SQL for non-decomposable recomputation
    pub base_relation_sql: Option<String>,
    /// Dimension alias -> expression mapping from the source view
    pub dimension_exprs: HashMap<String, String>,
    /// GROUP BY columns from the source view definition
    pub view_group_by_cols: Vec<String>,
    /// The original measure expression (e.g., "COUNT(DISTINCT user_id)")
    pub expression: String,
}

/// Look up which view contains a measure and return resolved measure info
/// Prioritizes default_table (the query's FROM table), then searches other views for JOINs
fn resolve_measure_source(measure_name: &str, default_table: &str) -> ResolvedMeasure {
    let views = MEASURE_VIEWS.lock().unwrap();

    // Helper to build ResolvedMeasure from a found measure
    let build_resolved = |m: &ViewMeasure, v: &MeasureView, source_view: &str| -> ResolvedMeasure {
        let agg_fn = extract_agg_function(&m.expression);
        let derived_expr = if extract_aggregation_function(&m.expression).is_none() {
            let expanded = expand_derived_measure_expr(&m.expression, v);
            if expanded != m.expression {
                Some(expanded)
            } else {
                None
            }
        } else {
            None
        };

        ResolvedMeasure {
            agg_fn,
            source_view: source_view.to_string(),
            derived_expr,
            is_decomposable: m.is_decomposable,
            base_table: v.base_table.clone(),
            base_relation_sql: v.base_relation_sql.clone(),
            dimension_exprs: v.dimension_exprs.clone(),
            view_group_by_cols: v.group_by_cols.clone(),
            expression: m.expression.clone(),
        }
    };

    // First, check if the measure exists in the default table (query's primary table)
    if let Some(v) = views.get(default_table) {
        if let Some(m) = v
            .measures
            .iter()
            .find(|m| m.column_name.eq_ignore_ascii_case(measure_name))
        {
            return build_resolved(m, v, default_table);
        }
    }

    // If not found in default table, search other views (for JOIN support)
    for (view_name, v) in views.iter() {
        if let Some(m) = v
            .measures
            .iter()
            .find(|m| m.column_name.eq_ignore_ascii_case(measure_name))
        {
            return build_resolved(m, v, view_name);
        }
    }

    // Fallback: measure not found, return defaults
    ResolvedMeasure {
        agg_fn: "SUM".to_string(),
        source_view: default_table.to_string(),
        derived_expr: None,
        is_decomposable: true,
        base_table: None,
        base_relation_sql: None,
        dimension_exprs: HashMap::new(),
        view_group_by_cols: Vec::new(),
        expression: String::new(),
    }
}

/// Find the alias used for a view in the FROM clause
/// Returns the effective_name (alias) if the view is in the FROM clause
fn find_alias_for_view<'a>(from_info: &'a FromClauseInfo, view_name: &str) -> Option<&'a str> {
    from_info
        .tables
        .values()
        .find(|t| t.name.eq_ignore_ascii_case(view_name))
        .map(|t| t.effective_name.as_str())
}

// =============================================================================
// Core Functions - Non-Decomposable Measure Expansion
// =============================================================================

fn base_relation_for_subquery(base_relation_sql: &str) -> String {
    let trimmed = base_relation_sql.trim().trim_end_matches(';').trim();
    format!("({trimmed})")
}

fn correlation_exprs_for_dim(
    dim: &str,
    dimension_exprs: &HashMap<String, String>,
    outer_alias: Option<&str>,
) -> (String, String) {
    let dim_trim = dim.trim();
    let dim_name = dim_trim.split('.').next_back().unwrap_or(dim_trim).trim();
    let dim_is_qualified = dim_trim.contains('.');
    let dim_key = normalize_dimension_key(dim_name);
    if let Some(expr) = dimension_exprs.get(&dim_key) {
        let inner_expr = qualify_where_for_inner(expr);
        let outer_expr = if dim_is_qualified {
            dim_trim.to_string()
        } else {
            outer_alias
                .map(|alias| format!("{alias}.{dim_name}"))
                .unwrap_or_else(|| dim_name.to_string())
        };
        return (inner_expr, outer_expr);
    }

    if dim_trim.contains('(') {
        let inner_expr = qualify_where_for_inner(dim_trim);
        let outer_expr = outer_alias
            .map(|alias| format!("ANY_VALUE({})", qualify_where_for_outer(dim_trim, alias)))
            .unwrap_or_else(|| dim_trim.to_string());
        return (inner_expr, outer_expr);
    }

    let inner_expr = format!("_inner.{dim_name}");
    let outer_expr = if dim_is_qualified {
        dim_trim.to_string()
    } else {
        outer_alias
            .map(|alias| format!("{alias}.{dim_name}"))
            .unwrap_or_else(|| dim_name.to_string())
    };
    (inner_expr, outer_expr)
}

fn correlation_condition_for_dim(
    dim: &str,
    dimension_exprs: &HashMap<String, String>,
    outer_alias: Option<&str>,
) -> String {
    let (inner_expr, outer_expr) = correlation_exprs_for_dim(dim, dimension_exprs, outer_alias);
    format!("{inner_expr} IS NOT DISTINCT FROM {outer_expr}")
}

struct NonDecompJoinPlan {
    join_sql: String,
    replacement: String,
}

fn expand_non_decomposable_default_context(
    expression: &str,
    base_relation_sql: &str,
    outer_alias: Option<&str>,
    group_by_cols: &[String],
    dimension_exprs: &HashMap<String, String>,
) -> String {
    let base_relation = base_relation_for_subquery(base_relation_sql);

    if group_by_cols.is_empty() {
        return format!("(SELECT {expression} FROM {base_relation})");
    }

    let where_clauses: Vec<_> = group_by_cols
        .iter()
        .map(|col| correlation_condition_for_dim(col, dimension_exprs, outer_alias))
        .collect();

    format!(
        "(SELECT {} FROM {} _inner WHERE {})",
        expression,
        base_relation,
        where_clauses.join(" AND ")
    )
}

fn build_non_decomposable_join_plan(
    expression: &str,
    base_relation_sql: &str,
    outer_alias: Option<&str>,
    outer_where: Option<&str>,
    group_by_cols: &[String],
    modifiers: &[ContextModifier],
    dimension_exprs: &HashMap<String, String>,
    join_alias: &str,
) -> Option<NonDecompJoinPlan> {
    let base_relation = base_relation_for_subquery(base_relation_sql);
    let mut removed_dims: Vec<String> = Vec::new();
    let mut effective_where: Option<String> = None;
    let mut set_overrides: HashMap<String, String> = HashMap::new();
    let mut has_all_global = false;

    let has_set = modifiers
        .iter()
        .any(|m| matches!(m, ContextModifier::Set(_, _)));

    if modifiers.is_empty() {
        if let Some(w) = outer_where {
            effective_where = Some(qualify_where_for_inner_with_dimensions(w, dimension_exprs));
        }
    } else {
        let all_are_all = modifiers
            .iter()
            .all(|m| matches!(m, ContextModifier::All(_) | ContextModifier::AllGlobal));
        if all_are_all {
            if modifiers
                .iter()
                .any(|m| matches!(m, ContextModifier::AllGlobal))
            {
                return None;
            }
            for modifier in modifiers {
                if let ContextModifier::All(dim) = modifier {
                    removed_dims.push(dim.to_lowercase());
                }
            }
        } else {
            for modifier in modifiers.iter().rev() {
                match modifier {
                    ContextModifier::AllGlobal => {
                        has_all_global = true;
                        effective_where = None;
                        set_overrides.clear();
                        removed_dims.clear();
                    }
                    ContextModifier::All(dim) => {
                        removed_dims.push(dim.to_lowercase());
                    }
                    ContextModifier::Visible => {
                        if !has_set && !has_all_global {
                            if let Some(w) = outer_where {
                                let stripped = strip_at_where_qualifiers(w);
                                effective_where = Some(qualify_where_for_inner_with_dimensions(
                                    &stripped,
                                    dimension_exprs,
                                ));
                            }
                        }
                    }
                    ContextModifier::Where(cond) => {
                        if !has_all_global {
                            let stripped = strip_at_where_qualifiers(cond);
                            effective_where =
                                Some(qualify_where_for_inner_with_dimensions(
                                    &stripped,
                                    dimension_exprs,
                                ));
                        }
                    }
                    ContextModifier::Set(dim, expr) => {
                        let dim_name = dim.split('.').next_back().unwrap_or(dim).trim();
                        let dim_key = normalize_dimension_key(dim_name);
                        if !has_all_global && !removed_dims.contains(&dim_key) {
                            let resolved_expr =
                                resolve_current_in_expr(expr, group_by_cols, outer_where, outer_alias);
                            let outer_expr = if let Some(alias) = outer_alias {
                                if dim.contains('(') {
                                    qualify_where_for_outer(&resolved_expr, alias)
                                } else {
                                    qualify_outer_reference(&resolved_expr, alias, dim_name)
                                }
                            } else {
                                resolved_expr
                            };
                            set_overrides.insert(dim_key, outer_expr);
                        }
                    }
                }
            }
        }
    }

    if has_all_global && set_overrides.is_empty() {
        return None;
    }

    let remaining_cols: Vec<&String> = group_by_cols
        .iter()
        .filter(|col| {
            let col_lower = col.to_lowercase();
            let col_name = col.split('.').next_back().unwrap_or(col).to_lowercase();
            !removed_dims
                .iter()
                .any(|d| *d == col_lower || *d == col_name)
        })
        .collect();

    if remaining_cols.is_empty() {
        return None;
    }

    let mut select_parts = Vec::new();
    let mut group_parts = Vec::new();
    let mut join_conditions = Vec::new();

    for (idx, col) in remaining_cols.iter().enumerate() {
        let alias = format!("dim_{idx}");
        let (inner_expr, default_outer_expr) =
            correlation_exprs_for_dim(col, dimension_exprs, outer_alias);
        let dim_name = col.split('.').next_back().unwrap_or(col).trim();
        let dim_key = normalize_dimension_key(dim_name);
        let outer_expr = set_overrides
            .get(&dim_key)
            .cloned()
            .unwrap_or(default_outer_expr);

        select_parts.push(format!("{inner_expr} AS {alias}"));
        group_parts.push(alias.clone());
        join_conditions.push(format!("{join_alias}.{alias} IS NOT DISTINCT FROM {outer_expr}"));
    }

    let mut agg_query = format!(
        "SELECT {}, {} AS value FROM {} _inner",
        select_parts.join(", "),
        expression,
        base_relation
    );
    if let Some(w) = effective_where {
        agg_query.push_str(" WHERE ");
        agg_query.push_str(&w);
    }
    if !group_parts.is_empty() {
        agg_query.push_str(" GROUP BY ");
        agg_query.push_str(&group_parts.join(", "));
    }

    Some(NonDecompJoinPlan {
        join_sql: format!(" LEFT JOIN ({agg_query}) {join_alias} ON {}", join_conditions.join(" AND ")),
        replacement: format!("{join_alias}.value"),
    })
}

/// Expand a non-decomposable measure (like COUNT DISTINCT) to a correlated subquery
/// against the base table. This is used when the measure cannot be re-aggregated.
///
/// Example:
/// - expression: "COUNT(DISTINCT user_id)"
/// - base_relation_sql: "SELECT * FROM orders"
/// - group_by_cols: ["year", "region"]
/// - Result: (SELECT COUNT(DISTINCT user_id) FROM (SELECT * FROM orders) _inner WHERE _inner.year IS NOT DISTINCT FROM _outer.year AND _inner.region IS NOT DISTINCT FROM _outer.region)
fn expand_non_decomposable_to_sql(
    expression: &str,
    base_relation_sql: &str,
    outer_alias: Option<&str>,
    outer_where: Option<&str>,
    group_by_cols: &[String],
    modifiers: &[ContextModifier],
    dimension_exprs: &HashMap<String, String>,
) -> String {
    let base_relation = base_relation_for_subquery(base_relation_sql);

    if modifiers.is_empty() {
        // No modifiers = default VISIBLE behavior (respect outer WHERE)
        return expand_non_decomposable_at_to_sql(
            expression,
            &ContextModifier::Visible,
            base_relation_sql,
            outer_alias,
            outer_where,
            group_by_cols,
            dimension_exprs,
        );
    }

    if modifiers.len() == 1 {
        return expand_non_decomposable_at_to_sql(
            expression,
            &modifiers[0],
            base_relation_sql,
            outer_alias,
            outer_where,
            group_by_cols,
            dimension_exprs,
        );
    }

    // Check if all modifiers are ALL (dimension or global)
    let all_are_all = modifiers
        .iter()
        .all(|m| matches!(m, ContextModifier::All(_) | ContextModifier::AllGlobal));
    if all_are_all {
        // Check for explicit AllGlobal - that means grand total
        if modifiers
            .iter()
            .any(|m| matches!(m, ContextModifier::AllGlobal))
        {
            return expand_non_decomposable_at_to_sql(
                expression,
                &ContextModifier::AllGlobal,
                base_relation_sql,
                outer_alias,
                outer_where,
                group_by_cols,
                dimension_exprs,
            );
        }

        // Accumulate all dimensions to remove
        let removed_dims: Vec<&str> = modifiers
            .iter()
            .filter_map(|m| match m {
                ContextModifier::All(dim) => Some(dim.as_str()),
                _ => None,
            })
            .collect();

        // Filter group_by_cols to get remaining dimensions
        let remaining_cols: Vec<&String> = group_by_cols
            .iter()
            .filter(|col| {
                let col_lower = col.to_lowercase();
                let col_name = col.split('.').next_back().unwrap_or(col).to_lowercase();
                !removed_dims
                    .iter()
                    .any(|d| d.to_lowercase() == col_lower || d.to_lowercase() == col_name)
            })
            .collect();

        if remaining_cols.is_empty() {
            // All dimensions removed = grand total
            return expand_non_decomposable_at_to_sql(
                expression,
                &ContextModifier::AllGlobal,
                base_relation_sql,
                outer_alias,
                outer_where,
                group_by_cols,
                dimension_exprs,
            );
        }

        // Generate correlation on remaining dimensions only
        let where_clauses: Vec<_> = remaining_cols
            .iter()
            .map(|col| correlation_condition_for_dim(col, dimension_exprs, outer_alias))
            .collect();
        return format!(
            "(SELECT {} FROM {} _inner WHERE {})",
            expression,
            base_relation,
            where_clauses.join(" AND ")
        );
    }

    // Apply modifiers right-to-left
    // - VISIBLE adds outer WHERE (but SET bypasses it per paper)
    // - ALL removes dimensions from correlation
    // - SET changes a dimension and bypasses outer WHERE
    // - WHERE adds a filter

    let has_set = modifiers
        .iter()
        .any(|m| matches!(m, ContextModifier::Set(_, _)));

    let mut effective_where: Option<String> = None;
    let mut has_all_global = false;
    let mut set_conditions: Vec<String> = Vec::new();
    let mut removed_dims: Vec<String> = Vec::new();

    for modifier in modifiers.iter().rev() {
        match modifier {
            ContextModifier::AllGlobal => {
                has_all_global = true;
                effective_where = None;
                set_conditions.clear();
            }
            ContextModifier::All(dim) => {
                removed_dims.push(dim.to_lowercase());
            }
            ContextModifier::Visible => {
                if !has_set && !has_all_global {
                    if let Some(w) = outer_where {
                        let stripped = strip_at_where_qualifiers(w);
                        effective_where = Some(qualify_where_for_inner_with_dimensions(
                            &stripped,
                            dimension_exprs,
                        ));
                    }
                }
            }
            ContextModifier::Where(cond) => {
                if !has_all_global {
                    let stripped = strip_at_where_qualifiers(cond);
                    effective_where =
                        Some(qualify_where_for_inner_with_dimensions(
                            &stripped,
                            dimension_exprs,
                        ));
                }
            }
            ContextModifier::Set(dim, expr) => {
                let dim_lower = dim.to_lowercase();
                if !has_all_global && !removed_dims.contains(&dim_lower) {
                    let outer_ref = outer_alias.unwrap_or("_outer");
                    let dim_name = dim.split('.').next_back().unwrap_or(dim).trim();
                    let dim_key = normalize_dimension_key(dim_name);
                    let resolved_expr = resolve_current_in_expr(
                        expr,
                        group_by_cols,
                        outer_where,
                        Some(outer_ref),
                    );
                    let qualified_expr = if dim.contains('(') {
                        qualify_where_for_outer(&resolved_expr, outer_ref)
                    } else {
                        qualify_outer_reference(&resolved_expr, outer_ref, dim_name)
                    };
                    let inner_dim = if let Some(expr) = dimension_exprs.get(&dim_key) {
                        qualify_where_for_inner(expr)
                    } else if dim.contains('(') {
                        qualify_where_for_inner(dim)
                    } else {
                        format!("_inner.{dim_name}")
                    };
                    set_conditions.push(format!("{inner_dim} IS NOT DISTINCT FROM {qualified_expr}"));
                }
            }
        }
    }

    if has_all_global && set_conditions.is_empty() {
        return format!("(SELECT {expression} FROM {base_relation})");
    }

    // Filter group_by_cols to exclude removed dimensions
    let remaining_cols: Vec<&String> = group_by_cols
        .iter()
        .filter(|col| {
            let col_lower = col.to_lowercase();
            let col_name = col.split('.').next_back().unwrap_or(col).to_lowercase();
            !removed_dims
                .iter()
                .any(|d| *d == col_lower || *d == col_name)
        })
        .collect();

    // Build correlation conditions for remaining dimensions
    let correlation_conditions: Vec<String> = remaining_cols
        .iter()
        .map(|col| correlation_condition_for_dim(col, dimension_exprs, outer_alias))
        .collect();

    // Combine all conditions: correlation + SET conditions + WHERE
    let mut all_conditions: Vec<String> = correlation_conditions;
    all_conditions.extend(set_conditions);
    if let Some(w) = effective_where {
        all_conditions.push(w);
    }

    if all_conditions.is_empty() {
        format!("(SELECT {expression} FROM {base_relation})")
    } else {
        format!(
            "(SELECT {} FROM {} _inner WHERE {})",
            expression,
            base_relation,
            all_conditions.join(" AND ")
        )
    }
}

/// Expand a single AT modifier for non-decomposable measures
fn expand_non_decomposable_at_to_sql(
    expression: &str,
    modifier: &ContextModifier,
    base_relation_sql: &str,
    outer_alias: Option<&str>,
    outer_where: Option<&str>,
    group_by_cols: &[String],
    dimension_exprs: &HashMap<String, String>,
) -> String {
    let base_relation = base_relation_for_subquery(base_relation_sql);

    match modifier {
        ContextModifier::AllGlobal => {
            format!("(SELECT {expression} FROM {base_relation})")
        }
        ContextModifier::All(dim) => {
            let dim_lower = dim.to_lowercase();
            let is_expression = dim.contains('(');
            let correlating_dims: Vec<_> = group_by_cols
                .iter()
                .filter(|col| {
                    if is_expression {
                        col.to_lowercase() != dim_lower
                    } else {
                        let col_name = col.split('.').next_back().unwrap_or(col);
                        col_name.to_lowercase() != dim_lower
                    }
                })
                .collect();

            if correlating_dims.is_empty() {
                format!("(SELECT {expression} FROM {base_relation})")
            } else {
                let where_clauses: Vec<_> = correlating_dims
                    .iter()
                    .map(|col| correlation_condition_for_dim(col, dimension_exprs, outer_alias))
                    .collect();
                format!(
                    "(SELECT {} FROM {} _inner WHERE {})",
                    expression,
                    base_relation,
                    where_clauses.join(" AND ")
                )
            }
        }
        ContextModifier::Set(dim, expr) => {
            let outer_ref = outer_alias.unwrap_or("_outer");
            let dim_name = dim.split('.').next_back().unwrap_or(dim).trim();
            let dim_key = normalize_dimension_key(dim_name);
            let resolved_expr =
                resolve_current_in_expr(expr, group_by_cols, outer_where, Some(outer_ref));
            let qualified_expr = if dim.contains('(') {
                qualify_where_for_outer(&resolved_expr, outer_ref)
            } else {
                qualify_outer_reference(&resolved_expr, outer_ref, dim_name)
            };
            let inner_dim = if let Some(expr) = dimension_exprs.get(&dim_key) {
                qualify_where_for_inner(expr)
            } else if dim.contains('(') {
                qualify_where_for_inner(dim)
            } else {
                format!("_inner.{dim_name}")
            };
            let set_condition = format!("{inner_dim} IS NOT DISTINCT FROM {qualified_expr}");

            let dim_lower = dim.to_lowercase();
            let is_expression = dim.contains('(');
            let correlation_conditions: Vec<String> = group_by_cols
                .iter()
                .filter(|col| {
                    if is_expression {
                        col.to_lowercase() != dim_lower
                    } else {
                        let col_name = col.split('.').next_back().unwrap_or(col);
                        col_name.to_lowercase() != dim_lower
                    }
                })
                .map(|col| correlation_condition_for_dim(col, dimension_exprs, outer_alias))
                .collect();

            let mut all_conditions = vec![set_condition];
            all_conditions.extend(correlation_conditions);

            format!(
                "(SELECT {} FROM {} _inner WHERE {})",
                expression,
                base_relation,
                all_conditions.join(" AND ")
            )
        }
        ContextModifier::Where(condition) => {
            let stripped = strip_at_where_qualifiers(condition);
            let qualified =
                qualify_where_for_inner_with_dimensions(&stripped, dimension_exprs);
            format!(
                "(SELECT {expression} FROM {base_relation} _inner WHERE {qualified})"
            )
        }
        ContextModifier::Visible => {
            if group_by_cols.is_empty() {
                match outer_where {
                    Some(w) => {
                        let stripped = strip_at_where_qualifiers(w);
                        let qualified =
                            qualify_where_for_inner_with_dimensions(&stripped, dimension_exprs);
                        format!(
                            "(SELECT {expression} FROM {base_relation} _inner WHERE {qualified})"
                        )
                    }
                    None => format!("(SELECT {expression} FROM {base_relation})"),
                }
            } else {
                let where_clauses: Vec<_> = group_by_cols
                    .iter()
                    .map(|col| correlation_condition_for_dim(col, dimension_exprs, outer_alias))
                    .collect();
                let full_where = match outer_where {
                    Some(w) => {
                        let stripped = strip_at_where_qualifiers(w);
                        let qualified = qualify_where_for_inner_with_dimensions(
                            &stripped,
                            dimension_exprs,
                        );
                        format!("{} AND {}", where_clauses.join(" AND "), qualified)
                    }
                    None => where_clauses.join(" AND "),
                };
                format!(
                    "(SELECT {} FROM {} _inner WHERE {full_where})",
                    expression, base_relation
                )
            }
        }
    }
}

// =============================================================================
// Core Functions - AT Expansion
// =============================================================================

/// Expand AT modifier to SQL subquery
/// When outer_alias is provided, it's used for correlated references in SET/ALL modifiers
/// When outer_where is provided, it's used for VISIBLE expansion
/// When group_by_cols is provided, it's used for ALL(dim) to correlate on other dimensions
pub fn expand_at_to_sql(
    measure_col: &str,
    agg_fn: &str,
    modifier: &ContextModifier,
    table_name: &str,
    outer_alias: Option<&str>,
    outer_where: Option<&str>,
    group_by_cols: &[String],
) -> String {
    let measure_expr = format!("{agg_fn}({measure_col})");

    match modifier {
        ContextModifier::AllGlobal => {
            // Grand total - no WHERE clause, aggregate over entire table
            format!("(SELECT {measure_expr} FROM {table_name})")
        }
        ContextModifier::All(dim) => {
            // Remove dimension from context - correlate on other GROUP BY dimensions
            let outer_ref = outer_alias.unwrap_or(table_name);
            // Filter group_by_cols to exclude the removed dimension (case-insensitive)
            let dim_lower = dim.to_lowercase();
            let is_expression = dim.contains('(');
            let correlating_dims: Vec<_> = group_by_cols
                .iter()
                .filter(|col| {
                    if is_expression {
                        // For expressions like MONTH(date), compare full expression
                        col.to_lowercase() != dim_lower
                    } else {
                        // For simple columns, extract just the column name (handle qualified refs like "s.year")
                        let col_name = col.split('.').next_back().unwrap_or(col);
                        col_name.to_lowercase() != dim_lower
                    }
                })
                .collect();

            if correlating_dims.is_empty() {
                // No other dimensions - same as AllGlobal
                format!("(SELECT {measure_expr} FROM {table_name})")
            } else {
                // Correlate on remaining dimensions
                let where_clauses: Vec<_> = correlating_dims
                    .iter()
                    .map(|col| {
                        let col_is_expr = col.contains('(');
                        if col_is_expr {
                            // For expression dimensions, qualify column refs inside
                            let inner_expr = qualify_where_for_inner(col);
                            format!("{inner_expr} IS NOT DISTINCT FROM {col}")
                        } else {
                            // Extract just the column name for _inner reference
                            let col_name = col.split('.').next_back().unwrap_or(col);
                            format!("_inner.{col_name} IS NOT DISTINCT FROM {outer_ref}.{col_name}")
                        }
                    })
                    .collect();
                format!(
                    "(SELECT {} FROM {} _inner WHERE {})",
                    measure_expr,
                    table_name,
                    where_clauses.join(" AND ")
                )
            }
        }
        ContextModifier::Set(dim, expr) => {
            // Use outer_alias for the correlated reference, falling back to table_name
            let outer_ref = outer_alias.unwrap_or(table_name);
            let dim_name = dim.split('.').next_back().unwrap_or(dim).trim();
            let resolved_expr =
                resolve_current_in_expr(expr, group_by_cols, outer_where, Some(outer_ref));
            let qualified_expr = if dim.contains('(') {
                qualify_where_for_outer(&resolved_expr, outer_ref)
            } else {
                qualify_outer_reference(&resolved_expr, outer_ref, dim_name)
            };
            // For ad hoc dimensions (expressions like MONTH(date)), qualify column refs with _inner
            // For simple columns, just prefix with _inner.
            let inner_dim = if dim.contains('(') {
                // Expression: qualify column refs inside it
                qualify_where_for_inner(dim)
            } else {
                format!("_inner.{dim_name}")
            };

            // SET condition for the specified dimension
            let set_condition = format!("{inner_dim} IS NOT DISTINCT FROM {qualified_expr}");

            // Build correlation conditions for OTHER GROUP BY columns (not the SET dim)
            // Per paper: SET only removes terms for the specified dimension, correlates on others
            let dim_lower = dim.to_lowercase();
            let is_expression = dim.contains('(');
            let correlation_conditions: Vec<String> = group_by_cols
                .iter()
                .filter(|col| {
                    if is_expression {
                        col.to_lowercase() != dim_lower
                    } else {
                        let col_name = col.split('.').next_back().unwrap_or(col);
                        col_name.to_lowercase() != dim_lower
                    }
                })
                .map(|col| {
                    let col_is_expr = col.contains('(');
                    if col_is_expr {
                        let inner_expr = qualify_where_for_inner(col);
                        format!("{inner_expr} IS NOT DISTINCT FROM {col}")
                    } else {
                        let col_name = col.split('.').next_back().unwrap_or(col);
                        format!("_inner.{col_name} IS NOT DISTINCT FROM {outer_ref}.{col_name}")
                    }
                })
                .collect();

            // Combine: SET condition + correlation on other dims
            // NOTE: Do NOT include outer_where - SET bypasses outer WHERE per paper
            let mut all_conditions = vec![set_condition];
            all_conditions.extend(correlation_conditions);

            format!(
                "(SELECT {} FROM {} _inner WHERE {})",
                measure_expr,
                table_name,
                all_conditions.join(" AND ")
            )
        }
        ContextModifier::Where(condition) => {
            let stripped = strip_at_where_qualifiers(condition);
            format!(
                "(SELECT {measure_expr} FROM {table_name} WHERE {stripped})"
            )
        }
        ContextModifier::Visible => {
            // VISIBLE means include outer query's WHERE clause AND respect GROUP BY context
            let outer_ref = outer_alias.unwrap_or(table_name);
            if group_by_cols.is_empty() {
                // No GROUP BY - just apply WHERE
                match outer_where {
                    Some(w) => format!("(SELECT {measure_expr} FROM {table_name} WHERE {w})"),
                    None => measure_expr,
                }
            } else {
                // Correlate on GROUP BY columns and apply WHERE
                let where_clauses: Vec<_> = group_by_cols
                    .iter()
                    .map(|col| {
                        // Extract just the column name for _inner reference
                        let col_name = col.split('.').next_back().unwrap_or(col);
                        format!("_inner.{col_name} IS NOT DISTINCT FROM {outer_ref}.{col_name}")
                    })
                    .collect();
                let full_where = match outer_where {
                    Some(w) => format!("{} AND {}", where_clauses.join(" AND "), w),
                    None => where_clauses.join(" AND "),
                };
                format!(
                    "(SELECT {measure_expr} FROM {table_name} _inner WHERE {full_where})"
                )
            }
        }
    }
}

/// Expand multiple AT modifiers sequentially (right-to-left per paper spec)
pub fn expand_modifiers_to_sql(
    measure_col: &str,
    agg_fn: &str,
    modifiers: &[ContextModifier],
    table_name: &str,
    outer_alias: Option<&str>,
    outer_where: Option<&str>,
    group_by_cols: &[String],
) -> String {
    if modifiers.is_empty() {
        // No modifiers = default VISIBLE behavior (respect outer WHERE)
        return expand_at_to_sql(
            measure_col,
            agg_fn,
            &ContextModifier::Visible,
            table_name,
            outer_alias,
            outer_where,
            group_by_cols,
        );
    }

    if modifiers.len() == 1 {
        return expand_at_to_sql(
            measure_col,
            agg_fn,
            &modifiers[0],
            table_name,
            outer_alias,
            outer_where,
            group_by_cols,
        );
    }

    // Check if all modifiers are ALL (dimension or global)
    let all_are_all = modifiers
        .iter()
        .all(|m| matches!(m, ContextModifier::All(_) | ContextModifier::AllGlobal));
    if all_are_all {
        // Check for explicit AllGlobal - that means grand total
        if modifiers
            .iter()
            .any(|m| matches!(m, ContextModifier::AllGlobal))
        {
            return expand_at_to_sql(
                measure_col,
                agg_fn,
                &ContextModifier::AllGlobal,
                table_name,
                outer_alias,
                outer_where,
                group_by_cols,
            );
        }

        // Accumulate all dimensions to remove
        let removed_dims: Vec<&str> = modifiers
            .iter()
            .filter_map(|m| match m {
                ContextModifier::All(dim) => Some(dim.as_str()),
                _ => None,
            })
            .collect();

        // Filter group_by_cols to get remaining dimensions
        let remaining_cols: Vec<&String> = group_by_cols
            .iter()
            .filter(|col| {
                let col_name = col.split('.').next_back().unwrap_or(col).to_lowercase();
                !removed_dims.iter().any(|d| d.to_lowercase() == col_name)
            })
            .collect();

        if remaining_cols.is_empty() {
            // All dimensions removed = grand total
            return expand_at_to_sql(
                measure_col,
                agg_fn,
                &ContextModifier::AllGlobal,
                table_name,
                outer_alias,
                outer_where,
                group_by_cols,
            );
        }

        // Generate correlation on remaining dimensions only
        let outer_ref = outer_alias.unwrap_or(table_name);
        let measure_expr = format!("{agg_fn}({measure_col})");
        let where_clauses: Vec<_> = remaining_cols
            .iter()
            .map(|col| {
                let col_name = col.split('.').next_back().unwrap_or(col);
                format!("_inner.{col_name} IS NOT DISTINCT FROM {outer_ref}.{col_name}")
            })
            .collect();
        return format!(
            "(SELECT {} FROM {} _inner WHERE {})",
            measure_expr,
            table_name,
            where_clauses.join(" AND ")
        );
    }

    // Apply modifiers right-to-left
    // For now, collect the effects:
    // - VISIBLE adds outer WHERE (but SET bypasses it per paper)
    // - ALL removes all filters
    // - SET changes a dimension and bypasses outer WHERE
    // - WHERE adds a filter

    // Check if SET is present - per paper, SET bypasses outer WHERE
    let has_set = modifiers
        .iter()
        .any(|m| matches!(m, ContextModifier::Set(_, _)));

    let mut effective_where: Option<String> = None;
    let mut has_all_global = false;
    let mut set_conditions: Vec<String> = Vec::new();
    let mut removed_dims: Vec<String> = Vec::new();

    // Process modifiers (in order, which is right-to-left per paper)
    for modifier in modifiers.iter().rev() {
        match modifier {
            ContextModifier::AllGlobal => {
                has_all_global = true;
                effective_where = None;
                set_conditions.clear();
            }
            ContextModifier::All(dim) => {
                // ALL dim removes that dimension from context
                // Track which dimensions are removed for later filtering
                removed_dims.push(dim.to_lowercase());
            }
            ContextModifier::Visible => {
                // Per paper: SET bypasses outer WHERE, so VISIBLE has no effect when SET is present
                if !has_set && !has_all_global {
                    if let Some(w) = outer_where {
                        let stripped = strip_at_where_qualifiers(w);
                        effective_where = Some(qualify_where_for_inner(&stripped));
                    }
                }
            }
            ContextModifier::Where(cond) => {
                if !has_all_global {
                    let stripped = strip_at_where_qualifiers(cond);
                    // Qualify column references with _inner
                    effective_where = Some(qualify_where_for_inner(&stripped));
                }
            }
            ContextModifier::Set(dim, expr) => {
                // Skip SET if ALL(dim) was already processed (dimension removed from context)
                let dim_lower = dim.to_lowercase();
                if !has_all_global && !removed_dims.contains(&dim_lower) {
                    let outer_ref = outer_alias.unwrap_or(table_name);
                    let dim_name = dim.split('.').next_back().unwrap_or(dim).trim();
                    let resolved_expr = resolve_current_in_expr(
                        expr,
                        group_by_cols,
                        outer_where,
                        Some(outer_ref),
                    );
                    let qualified_expr = if dim.contains('(') {
                        qualify_where_for_outer(&resolved_expr, outer_ref)
                    } else {
                        qualify_outer_reference(&resolved_expr, outer_ref, dim_name)
                    };
                    // For ad hoc dimensions (expressions), qualify column refs inside
                    let inner_dim = if dim.contains('(') {
                        qualify_where_for_inner(dim)
                    } else {
                        format!("_inner.{dim_name}")
                    };
                    set_conditions.push(format!("{inner_dim} IS NOT DISTINCT FROM {qualified_expr}"));
                }
            }
        }
    }

    // Build final SQL
    let measure_expr = format!("{agg_fn}({measure_col})");

    if has_all_global && set_conditions.is_empty() {
        // Pure grand total
        return format!("(SELECT {measure_expr} FROM {table_name})");
    }

    // Filter group_by_cols to exclude removed dimensions
    let remaining_cols: Vec<&String> = group_by_cols
        .iter()
        .filter(|col| {
            let col_lower = col.to_lowercase();
            let col_name = col.split('.').next_back().unwrap_or(col).to_lowercase();
            // Check both full expression match and simple column match
            !removed_dims
                .iter()
                .any(|d| *d == col_lower || *d == col_name)
        })
        .collect();

    // Build correlation conditions for remaining dimensions
    let outer_ref = outer_alias.unwrap_or(table_name);
    let correlation_conditions: Vec<String> = remaining_cols
        .iter()
        .map(|col| {
            let col_is_expr = col.contains('(');
            if col_is_expr {
                // For expression dimensions, qualify column refs inside
                let inner_expr = qualify_where_for_inner(col);
                format!("{inner_expr} IS NOT DISTINCT FROM {col}")
            } else {
                let col_name = col.split('.').next_back().unwrap_or(col);
                format!("_inner.{col_name} IS NOT DISTINCT FROM {outer_ref}.{col_name}")
            }
        })
        .collect();

    // Combine all conditions: correlation + SET conditions + WHERE
    let mut all_conditions: Vec<String> = correlation_conditions;
    all_conditions.extend(set_conditions);
    if let Some(w) = effective_where {
        all_conditions.push(w);
    }

    if all_conditions.is_empty() {
        format!("(SELECT {measure_expr} FROM {table_name})")
    } else {
        format!(
            "(SELECT {} FROM {} _inner WHERE {})",
            measure_expr,
            table_name,
            all_conditions.join(" AND ")
        )
    }
}

/// Expand AT modifiers for derived measures (pre-expanded expression like "SUM(revenue) - SUM(cost)")
fn expand_modifiers_to_sql_derived(
    derived_expr: &str,
    modifiers: &[ContextModifier],
    table_name: &str,
    outer_alias: Option<&str>,
    outer_where: Option<&str>,
    group_by_cols: &[String],
) -> String {
    // For derived measures, we use the pre-expanded expression directly
    // The logic is similar to expand_modifiers_to_sql but uses derived_expr instead of AGG(measure)

    if modifiers.is_empty() {
        // No modifiers = just use the expression
        return format!("(SELECT {derived_expr} FROM {table_name})");
    }

    // Check for AllGlobal (grand total)
    if modifiers
        .iter()
        .any(|m| matches!(m, ContextModifier::AllGlobal))
    {
        return format!("(SELECT {derived_expr} FROM {table_name})");
    }

    // Check if all modifiers are ALL (dimension)
    let all_are_all = modifiers
        .iter()
        .all(|m| matches!(m, ContextModifier::All(_)));
    if all_are_all {
        // Accumulate dimensions to remove
        let removed_dims: Vec<&str> = modifiers
            .iter()
            .filter_map(|m| match m {
                ContextModifier::All(dim) => Some(dim.as_str()),
                _ => None,
            })
            .collect();

        // Filter group_by_cols to get remaining dimensions
        let remaining_cols: Vec<&String> = group_by_cols
            .iter()
            .filter(|col| {
                let col_lower = col.to_lowercase();
                let col_name = col.split('.').next_back().unwrap_or(col).to_lowercase();
                // Check both full expression match and simple column match
                !removed_dims
                    .iter()
                    .any(|d| d.to_lowercase() == col_lower || d.to_lowercase() == col_name)
            })
            .collect();

        if remaining_cols.is_empty() {
            // All dimensions removed = grand total
            return format!("(SELECT {derived_expr} FROM {table_name})");
        }

        // Generate correlation on remaining dimensions
        let outer_ref = outer_alias.unwrap_or(table_name);
        let where_clauses: Vec<_> = remaining_cols
            .iter()
            .map(|col| {
                let col_is_expr = col.contains('(');
                if col_is_expr {
                    // For expression dimensions, qualify column refs inside
                    let inner_expr = qualify_where_for_inner(col);
                    format!("{inner_expr} IS NOT DISTINCT FROM {col}")
                } else {
                    let col_name = col.split('.').next_back().unwrap_or(col);
                    format!("_inner.{col_name} IS NOT DISTINCT FROM {outer_ref}.{col_name}")
                }
            })
            .collect();
        return format!(
            "(SELECT {} FROM {} _inner WHERE {})",
            derived_expr,
            table_name,
            where_clauses.join(" AND ")
        );
    }

    // For other modifiers (SET, WHERE, VISIBLE), build conditions
    // Check if SET is present - per paper, SET bypasses outer WHERE
    let has_set = modifiers
        .iter()
        .any(|m| matches!(m, ContextModifier::Set(_, _)));

    let mut effective_where: Option<String> = None;
    let mut has_all_global = false;
    let mut removed_dims: Vec<String> = Vec::new();

    for modifier in modifiers.iter().rev() {
        match modifier {
            ContextModifier::AllGlobal => {
                has_all_global = true;
                effective_where = None;
            }
            ContextModifier::All(dim) => {
                removed_dims.push(dim.to_lowercase());
            }
            ContextModifier::Visible => {
                // Per paper: SET bypasses outer WHERE, so VISIBLE has no effect when SET is present
                if !has_set && !has_all_global {
                    if let Some(w) = outer_where {
                        let stripped = strip_at_where_qualifiers(w);
                        effective_where = Some(qualify_where_for_inner(&stripped));
                    }
                }
            }
            ContextModifier::Where(cond) => {
                if !has_all_global {
                    let stripped = strip_at_where_qualifiers(cond);
                    effective_where = Some(qualify_where_for_inner(&stripped));
                }
            }
            ContextModifier::Set(_, _) => {
                // SET doesn't apply directly to derived measures in the same way
                // The derived expression already includes the aggregations
            }
        }
    }

    if has_all_global {
        return format!("(SELECT {derived_expr} FROM {table_name})");
    }

    // Filter remaining dimensions
    let remaining_cols: Vec<&String> = group_by_cols
        .iter()
        .filter(|col| {
            let col_lower = col.to_lowercase();
            let col_name = col.split('.').next_back().unwrap_or(col).to_lowercase();
            // Check both full expression match and simple column match
            !removed_dims
                .iter()
                .any(|d| *d == col_lower || *d == col_name)
        })
        .collect();

    // Build conditions
    let outer_ref = outer_alias.unwrap_or(table_name);
    let mut all_conditions: Vec<String> = remaining_cols
        .iter()
        .map(|col| {
            let col_is_expr = col.contains('(');
            if col_is_expr {
                // For expression dimensions, qualify column refs inside
                let inner_expr = qualify_where_for_inner(col);
                format!("{inner_expr} IS NOT DISTINCT FROM {col}")
            } else {
                let col_name = col.split('.').next_back().unwrap_or(col);
                format!("_inner.{col_name} IS NOT DISTINCT FROM {outer_ref}.{col_name}")
            }
        })
        .collect();

    if let Some(w) = effective_where {
        all_conditions.push(w);
    }

    if all_conditions.is_empty() {
        format!("(SELECT {derived_expr} FROM {table_name})")
    } else {
        format!(
            "(SELECT {} FROM {} _inner WHERE {})",
            derived_expr,
            table_name,
            all_conditions.join(" AND ")
        )
    }
}

fn validate_set_expression_requirements(
    at_patterns: &[(String, Vec<ContextModifier>, usize, usize)],
    group_by_cols: &[String],
    default_qualifier: Option<&str>,
) -> Option<String> {
    for (_, modifiers, _, _) in at_patterns {
        for modifier in modifiers {
            if let ContextModifier::Set(dim, expr) = modifier {
                if dim.contains('(') {
                    continue;
                }
                let dim_name = dim.split('.').next_back().unwrap_or(dim).trim();
                if expr_mentions_identifier_outside_current(expr, dim_name)
                    && !dimension_in_group_by(dim, group_by_cols, default_qualifier)
                {
                    return Some(format!(
                        "AT (SET {dim} = {expr}) references {dim_name}, but the query does not group by {dim_name}. Add {dim_name} to SELECT/GROUP BY or use a constant SET value."
                    ));
                }
            }
        }
    }

    None
}

/// Expand AGGREGATE() with AT modifiers in SQL
pub fn expand_aggregate_with_at(sql: &str) -> AggregateExpandResult {
    let cte_expansion = expand_cte_queries(sql);
    let mut sql = cte_expansion.sql;
    let mut had_aggregate = cte_expansion.had_aggregate;

    if has_measure_at_refs(&sql) {
        sql = rewrite_measure_at_refs(&sql);
        had_aggregate = true;
    }

    if has_implicit_measure_refs(&sql) {
        sql = rewrite_implicit_measure_refs(&sql);
        had_aggregate = true;
    }

    // Check if we need the full expansion path (AT modifiers or non-decomposable measures)
    let has_aggregate = has_aggregate_function(&sql);

    // If no AGGREGATE function at all, nothing to do
    if !has_aggregate {
        return AggregateExpandResult {
            had_aggregate,
            expanded_sql: sql,
            error: None,
        };
    }
    had_aggregate = true;

    let at_patterns = extract_aggregate_with_at_full(&sql);
    // Keep full expansion path even without AT to handle non-decomposable measures safely

    // Prefer parser-FFI FROM extraction (supports JOIN aliases when SQL parses there),
    // then fall back to string extraction for AGGREGATE/AT syntax that parser-FFI cannot parse.
    let mut from_info = extract_from_clause_info_ffi(&sql);
    let (primary_table_name, existing_alias) = if let Some(pt) = from_info.primary_table.clone() {
        let alias = if pt.has_alias {
            Some(pt.effective_name.clone())
        } else {
            None
        };
        (pt.name, alias)
    } else {
        let (table_name, alias) =
            extract_table_and_alias_from_sql(&sql).unwrap_or_else(|| ("t".to_string(), None));
        let primary_table = TableInfo {
            name: table_name.clone(),
            effective_name: alias.clone().unwrap_or_else(|| table_name.clone()),
            has_alias: alias.is_some(),
        };
        from_info
            .tables
            .insert(primary_table.effective_name.clone(), primary_table.clone());
        from_info.primary_table = Some(primary_table);
        (table_name, alias)
    };

    // Extract outer WHERE clause for VISIBLE semantics
    let outer_where = extract_where_clause(&sql);
    let outer_where_ref = outer_where.as_deref();

    // Extract GROUP BY columns for AT (ALL dim) correlation
    let group_by_cols = extract_group_by_columns(&sql);

    let default_set_qualifier = if let Some(alias) = existing_alias.as_deref() {
        Some(alias)
    } else if primary_table_name.is_empty() {
        None
    } else {
        Some(primary_table_name.as_str())
    };
    if let Some(error) = validate_set_expression_requirements(
        &at_patterns,
        &group_by_cols,
        default_set_qualifier,
    ) {
        return AggregateExpandResult {
            had_aggregate: true,
            expanded_sql: sql.to_string(),
            error: Some(error),
        };
    }

    // Extract dimension columns from original SQL for implicit GROUP BY
    // (must be done before expansion since expanded SQL has SUM() etc)
    let original_dim_cols = extract_dimension_columns_from_select(&sql);
    let effective_group_by_cols = if group_by_cols.is_empty() {
        original_dim_cols.clone()
    } else {
        group_by_cols.clone()
    };

    let has_expression_dimensions = original_dim_cols.iter().any(|col| col.contains('('));

    // Check if query needs an explicit outer alias for correlation handling.
    let needs_outer_alias = has_expression_dimensions
        || at_patterns.iter().any(|(_, modifiers, _, _)| {
        modifiers.iter().any(|m| {
            matches!(m, ContextModifier::Set(_, _))
                || matches!(m, ContextModifier::All(_))
                || matches!(m, ContextModifier::Visible)
        })
    });

    let mut result_sql = sql;

    // Handle alias for the primary table if needed for correlation
    let mut insert_outer_alias = false;
    let primary_alias: Option<String> = if needs_outer_alias {
        if let Some(ref pt) = from_info.primary_table {
            if pt.has_alias {
                Some(pt.effective_name.clone())
            } else if insert_primary_table_alias(&result_sql, "_outer").is_some() {
                insert_outer_alias = true;
                Some("_outer".to_string())
            } else {
                None
            }
        } else {
            None
        }
    } else {
        None
    };

    let mut patterns = at_patterns;
    patterns.sort_by(|a, b| b.2.cmp(&a.2));

    for (measure_name, modifiers, start, end) in patterns {
        let measure_lookup_name = strip_measure_qualifier(&measure_name);
        // Look up which view contains this measure (for JOIN support)
        let resolved = resolve_measure_source(&measure_lookup_name, &primary_table_name);
        let mut measure_group_by_cols = filter_group_by_cols_for_measure(
            &effective_group_by_cols,
            &resolved.view_group_by_cols,
            &resolved.dimension_exprs,
        );
        let mut allowed_qualifiers: HashSet<String> = HashSet::new();
        allowed_qualifiers.insert(normalize_identifier_name(&resolved.source_view));
        if let Some(alias) = find_alias_for_view(&from_info, &resolved.source_view) {
            allowed_qualifiers.insert(normalize_identifier_name(alias));
        }
        if let Some(ref pt) = from_info.primary_table {
            if pt.name.eq_ignore_ascii_case(&resolved.source_view) {
                if let Some(alias) = primary_alias.as_deref() {
                    allowed_qualifiers.insert(normalize_identifier_name(alias));
                }
            }
        }
        let source_dims = source_dimension_names(&resolved.source_view);
        measure_group_by_cols.retain(|col| {
            if let Some((Some(qualifier), dim_name)) = parse_simple_measure_ref(col) {
                return allowed_qualifiers.contains(&qualifier) || source_dims.contains(&dim_name);
            }
            let dim_key = normalize_dimension_key(col);
            source_dims.is_empty()
                || source_dims.contains(&dim_key)
                || resolved.dimension_exprs.contains_key(&dim_key)
                || source_dims
                    .iter()
                    .any(|dim_name| expr_mentions_identifier(col, dim_name))
        });
        let eval_group_by_cols =
            if measure_group_by_cols.is_empty()
                && !original_dim_cols.is_empty()
                && resolved.source_view.eq_ignore_ascii_case(&primary_table_name)
            {
            original_dim_cols.clone()
        } else {
            measure_group_by_cols.clone()
        };

        // Non-decomposable measures are recomputed from base rows (including AT modifiers)

        // Find the alias for this measure's source view in the FROM clause
        // If the source view is the primary table and we added _outer, use _outer
        let outer_alias = if let Some(ref pt) = from_info.primary_table {
            if pt.name.eq_ignore_ascii_case(&resolved.source_view) {
                // Source view is the primary table, use primary_alias (which may be _outer)
                primary_alias.clone()
            } else {
                // Source view is not the primary table, look it up in from_info
                find_alias_for_view(&from_info, &resolved.source_view)
                    .map(|s| s.to_string())
                    .or_else(|| primary_alias.clone())
            }
        } else {
            primary_alias.clone()
        };
        let outer_alias_ref = outer_alias.as_deref();

        let outer_ref_for_eval = outer_alias_ref.or(Some(resolved.source_view.as_str()));
        let base_relation_sql = resolved
            .base_relation_sql
            .clone()
            .or_else(|| {
                resolved
                    .base_table
                    .clone()
                    .map(|table| format!("SELECT * FROM {table}"))
            })
            .unwrap_or_else(|| format!("SELECT * FROM {}", resolved.source_view));

        let expression_for_eval = resolved
            .derived_expr
            .clone()
            .unwrap_or_else(|| resolved.expression.clone());

        let expanded = if !expression_for_eval.is_empty() {
            let eval_sql = expand_non_decomposable_to_sql(
                &expression_for_eval,
                &base_relation_sql,
                outer_ref_for_eval,
                outer_where_ref,
                &eval_group_by_cols,
                &modifiers,
                &resolved.dimension_exprs,
            );
            if original_dim_cols.is_empty() {
                format!("MAX({eval_sql})")
            } else {
                eval_sql
            }
        } else {
            expand_modifiers_to_sql(
                &measure_lookup_name,
                &resolved.agg_fn,
                &modifiers,
                &resolved.source_view,
                outer_alias_ref,
                outer_where_ref,
                &eval_group_by_cols,
            )
        };
        result_sql = format!("{}{}{}", &result_sql[..start], expanded, &result_sql[end..]);
    }

    // Also expand plain AGGREGATE() calls (without AT modifiers) using text replacement
    let mut plain_calls = extract_all_aggregate_calls(&result_sql);
    plain_calls.sort_by(|a, b| b.1.cmp(&a.1)); // Sort by position descending

    for (measure_name, start, end) in plain_calls {
        let mut replacement_end = end;
        let mut use_default_context = false;
        let suffix = &result_sql[end..];
        let leading_ws = suffix.len() - suffix.trim_start().len();
        if suffix[leading_ws..].starts_with(DEFAULT_CONTEXT_MARKER) {
            use_default_context = true;
            replacement_end = end + leading_ws + DEFAULT_CONTEXT_MARKER.len();
        }

        let measure_lookup_name = strip_measure_qualifier(&measure_name);
        let resolved = resolve_measure_source(&measure_lookup_name, &primary_table_name);
        let mut measure_group_by_cols = filter_group_by_cols_for_measure(
            &effective_group_by_cols,
            &resolved.view_group_by_cols,
            &resolved.dimension_exprs,
        );
        let mut allowed_qualifiers: HashSet<String> = HashSet::new();
        allowed_qualifiers.insert(normalize_identifier_name(&resolved.source_view));
        if let Some(alias) = find_alias_for_view(&from_info, &resolved.source_view) {
            allowed_qualifiers.insert(normalize_identifier_name(alias));
        }
        if let Some(ref pt) = from_info.primary_table {
            if pt.name.eq_ignore_ascii_case(&resolved.source_view) {
                if let Some(alias) = primary_alias.as_deref() {
                    allowed_qualifiers.insert(normalize_identifier_name(alias));
                }
            }
        }
        let source_dims = source_dimension_names(&resolved.source_view);
        measure_group_by_cols.retain(|col| {
            if let Some((Some(qualifier), dim_name)) = parse_simple_measure_ref(col) {
                return allowed_qualifiers.contains(&qualifier) || source_dims.contains(&dim_name);
            }
            let dim_key = normalize_dimension_key(col);
            source_dims.is_empty()
                || source_dims.contains(&dim_key)
                || resolved.dimension_exprs.contains_key(&dim_key)
                || source_dims
                    .iter()
                    .any(|dim_name| expr_mentions_identifier(col, dim_name))
        });
        let eval_group_by_cols =
            if measure_group_by_cols.is_empty()
                && !original_dim_cols.is_empty()
                && resolved.source_view.eq_ignore_ascii_case(&primary_table_name)
            {
            original_dim_cols.clone()
        } else {
            measure_group_by_cols.clone()
        };


        let outer_alias = if let Some(ref pt) = from_info.primary_table {
            if pt.name.eq_ignore_ascii_case(&resolved.source_view) {
                primary_alias.clone().or_else(|| {
                    find_alias_for_view(&from_info, &resolved.source_view).map(|s| s.to_string())
                })
            } else {
                find_alias_for_view(&from_info, &resolved.source_view)
                    .map(|s| s.to_string())
                    .or_else(|| primary_alias.clone())
            }
        } else {
            primary_alias.clone()
        };
        let outer_ref_for_eval = outer_alias.as_deref().or(Some(resolved.source_view.as_str()));
        let base_relation_sql = resolved
            .base_relation_sql
            .clone()
            .or_else(|| {
                resolved
                    .base_table
                    .clone()
                    .map(|table| format!("SELECT * FROM {table}"))
            })
            .unwrap_or_else(|| format!("SELECT * FROM {}", resolved.source_view));
        let expression_for_eval = resolved
            .derived_expr
            .clone()
            .unwrap_or_else(|| resolved.expression.clone());

        let expanded = if !expression_for_eval.is_empty() {
            let eval_sql = if use_default_context {
                expand_non_decomposable_default_context(
                    &expression_for_eval,
                    &base_relation_sql,
                    outer_ref_for_eval,
                    &eval_group_by_cols,
                    &resolved.dimension_exprs,
                )
            } else {
                expand_non_decomposable_to_sql(
                    &expression_for_eval,
                    &base_relation_sql,
                    outer_ref_for_eval,
                    outer_where_ref,
                    &eval_group_by_cols,
                    &[], // No modifiers for explicit AGGREGATE()
                    &resolved.dimension_exprs,
                )
            };
            if original_dim_cols.is_empty() {
                format!("MAX({eval_sql})")
            } else {
                eval_sql
            }
        } else {
            format!("{}({measure_lookup_name})", resolved.agg_fn)
        };
        result_sql = format!(
            "{}{}{}",
            &result_sql[..start],
            expanded,
            &result_sql[replacement_end..]
        );
    }

    if insert_outer_alias {
        if let Some(updated_sql) = insert_primary_table_alias(&result_sql, "_outer") {
            result_sql = updated_sql;
        }
    }

    // Check if there are still any remaining AGGREGATE calls (shouldn't be, but just in case)
    if has_aggregate_function(&result_sql) {
        return expand_aggregate(&result_sql);
    }

    // If no GROUP BY, add explicit GROUP BY with dimension columns from original SQL
    // (GROUP BY ALL doesn't work reliably with scalar subqueries mixed with aggregates)
    let result_upper = result_sql.to_uppercase();
    if !has_group_by_anywhere(&result_sql) && !original_dim_cols.is_empty() {
        // Find insertion point: before ORDER BY, LIMIT, HAVING, or at end
        let insert_pos = ["ORDER BY", "LIMIT", "HAVING", ";"]
            .iter()
            .filter_map(|kw| result_upper.find(kw))
            .min()
            .unwrap_or(result_sql.len());

        result_sql = format!(
            "{} GROUP BY {}{}",
            result_sql[..insert_pos].trim_end(),
            original_dim_cols.join(", "),
            if insert_pos < result_sql.len() {
                format!(" {}", result_sql[insert_pos..].trim_start())
            } else {
                String::new()
            }
        );
    }

    AggregateExpandResult {
        had_aggregate,
        expanded_sql: result_sql,
        error: None,
    }
}

// =============================================================================
// Catalog Functions
// =============================================================================

pub fn store_measure_view(
    view_name: &str,
    measures: Vec<ViewMeasure>,
    base_query: &str,
    base_table: Option<String>,
) {
    let view_query = extract_view_query(base_query).unwrap_or_else(|| base_query.to_string());
    let base_relation_sql = extract_base_relation_sql(&view_query);
    let dimension_exprs = extract_dimension_exprs_from_query(&view_query);
    let group_by_cols = extract_view_group_by_cols(&view_query);
    let measure_view = MeasureView {
        view_name: view_name.to_string(),
        measures,
        base_query: base_query.to_string(),
        base_table,
        base_relation_sql,
        dimension_exprs,
        group_by_cols,
    };

    let mut views = MEASURE_VIEWS.lock().unwrap();
    views.insert(view_name.to_string(), measure_view);
}

pub fn get_measure_view(view_name: &str) -> Option<MeasureView> {
    let views = MEASURE_VIEWS.lock().unwrap();
    views.get(view_name).cloned()
}

pub fn clear_measure_views() {
    let mut views = MEASURE_VIEWS.lock().unwrap();
    views.clear();
}

pub fn drop_measure_view(view_name: &str) -> bool {
    let mut views = MEASURE_VIEWS.lock().unwrap();
    let key = views
        .keys()
        .find(|k| k.eq_ignore_ascii_case(view_name))
        .cloned();
    if let Some(k) = key {
        views.remove(&k);
        return true;
    }
    false
}

pub fn drop_measure_view_from_sql(sql: &str) -> bool {
    let name = match extract_drop_view_name(sql) {
        Some(name) => name,
        None => return false,
    };
    drop_measure_view(&name)
}

pub fn get_measure_aggregation(column_name: &str) -> Option<(String, String)> {
    let views = MEASURE_VIEWS.lock().unwrap();

    for (view_name, view) in views.iter() {
        for measure in &view.measures {
            if measure.column_name.eq_ignore_ascii_case(column_name) {
                if let Some(agg_fn) = extract_aggregation_function(&measure.expression) {
                    return Some((agg_fn, view_name.clone()));
                }
            }
        }
    }
    None
}

fn extract_group_by_columns(sql: &str) -> Vec<String> {
    if let Ok(info) = parser_ffi::parse_select(sql) {
        if info.has_group_by {
            if info.group_by_all {
                return extract_dimension_columns_from_select(sql);
            }
            if !info.group_by_cols.is_empty() {
                return info
                    .group_by_cols
                    .into_iter()
                    .map(|c| c.trim().to_string())
                    .filter(|c| !c.is_empty() && !c.chars().all(|ch| ch.is_ascii_digit()))
                    .collect();
            }
        }
    }

    let mut columns = Vec::new();

    let query = sql.trim().trim_end_matches(';').trim();
    if let Some(group_by_pos) = find_top_level_keyword(query, "GROUP BY", 0) {
        let start = advance_after_group_by(query, group_by_pos)
            .unwrap_or_else(|| group_by_pos + "GROUP BY".len());
        let end = find_first_top_level_keyword(
            query,
            start,
            &["ORDER BY", "LIMIT", "HAVING", "QUALIFY", "WINDOW", "UNION", "INTERSECT", "EXCEPT"],
        )
        .unwrap_or(query.len());

        let group_by_content = query[start..end].trim();

        for part in group_by_content.split(',') {
            let col = part.trim();
            if !col.is_empty() && !col.chars().all(|c| c.is_ascii_digit()) {
                columns.push(col.to_string());
            }
        }
    }

    // If no GROUP BY, infer dimension columns from SELECT (non-AGGREGATE columns)
    if columns.is_empty() {
        columns = extract_dimension_columns_from_select(sql);
    }

    columns
}

/// Check if a SQL expression is a literal constant (integer, float, string, NULL, TRUE, FALSE,
/// or typed literals like DATE '...', TIMESTAMP '...', INTERVAL '...').
/// These should not be included in GROUP BY since they are not dimension columns.
fn is_literal_constant(expr: &str) -> bool {
    let trimmed = expr.trim();
    if trimmed.is_empty() {
        return false;
    }

    let upper = trimmed.to_uppercase();

    // NULL, TRUE, FALSE
    if upper == "NULL" || upper == "TRUE" || upper == "FALSE" {
        return true;
    }

    // String literal: starts and ends with single quote
    if trimmed.starts_with('\'') && trimmed.ends_with('\'') && trimmed.len() >= 2 {
        return true;
    }

    // Typed literals: DATE '...', TIMESTAMP '...', INTERVAL '...', TIME '...'
    for prefix in &["DATE ", "TIMESTAMP ", "INTERVAL ", "TIME "] {
        if upper.starts_with(prefix) {
            let rest = trimmed[prefix.len()..].trim();
            if rest.starts_with('\'') && rest.ends_with('\'') {
                return true;
            }
        }
    }

    // Numeric: integer or float (possibly negative)
    let numeric_part = if trimmed.starts_with('-') || trimmed.starts_with('+') {
        &trimmed[1..]
    } else {
        trimmed
    };

    if !numeric_part.is_empty() {
        let mut saw_dot = false;
        let is_numeric = numeric_part.chars().all(|c| {
            if c == '.' && !saw_dot {
                saw_dot = true;
                true
            } else {
                c.is_ascii_digit()
            }
        });
        // Must have at least one digit (not just "." or "-")
        if is_numeric && numeric_part.chars().any(|c| c.is_ascii_digit()) {
            return true;
        }
    }

    false
}

/// Extract non-AGGREGATE columns from SELECT clause to use as implicit GROUP BY columns
fn looks_like_sql_aggregate_expr(expr: &str) -> bool {
    // Prefer parser-backed detection when parser FFI is available.
    let parsed = std::panic::catch_unwind(|| parser_ffi::parse_expression(expr))
        .ok()
        .and_then(|result| result.ok());
    if let Some(info) = parsed {
        if info.is_aggregate {
            return true;
        }
    }

    // Fallback heuristic for environments where parser FFI is unavailable.
    let upper = expr.to_uppercase();
    [
        "COUNT(",
        "COUNT_STAR(",
        "SUM(",
        "AVG(",
        "MIN(",
        "MAX(",
        "ANY_VALUE(",
        "STRING_AGG(",
        "ARRAY_AGG(",
        "LIST(",
        "FIRST(",
        "LAST(",
        "MEDIAN(",
        "MODE(",
        "STDDEV(",
        "STDDEV_POP(",
        "STDDEV_SAMP(",
        "VAR_POP(",
        "VAR_SAMP(",
        "VARIANCE(",
        "QUANTILE(",
        "QUANTILE_CONT(",
        "QUANTILE_DISC(",
        "PERCENTILE_CONT(",
        "PERCENTILE_DISC(",
        "BOOL_AND(",
        "BOOL_OR(",
        "BIT_AND(",
        "BIT_OR(",
        "BIT_XOR(",
    ]
    .iter()
    .any(|pattern| upper.contains(pattern))
}

fn extract_dimension_columns_from_select(sql: &str) -> Vec<String> {
    let mut columns = Vec::new();

    let query = sql.trim().trim_end_matches(';').trim();
    let select_pos = match find_top_level_keyword(query, "SELECT", 0) {
        Some(pos) => pos,
        None => return columns,
    };
    let from_pos = match find_top_level_keyword(query, "FROM", select_pos) {
        Some(pos) => pos,
        None => return columns,
    };

    let select_start = select_pos + "SELECT".len();
    if select_start >= from_pos {
        return columns;
    }

    let select_content = &query[select_start..from_pos];

    // Split by comma, but be careful about nested parens
    let mut depth = 0;
    let mut current = String::new();
    let mut items = Vec::new();

    for c in select_content.chars() {
        match c {
            '(' => {
                depth += 1;
                current.push(c);
            }
            ')' => {
                depth -= 1;
                current.push(c);
            }
            ',' if depth == 0 => {
                items.push(current.trim().to_string());
                current = String::new();
            }
            _ => current.push(c),
        }
    }
    if !current.trim().is_empty() {
        items.push(current.trim().to_string());
    }

    // Filter out AGGREGATE() calls, literal constants, and extract column names
    for item in items {
        if has_aggregate_function(&item) {
            continue;
        }
        let item_upper = item.to_uppercase();
        // Handle "col AS alias" - use the column name, not alias
        let col = if let Some(as_pos) = item_upper.find(" AS ") {
            item[..as_pos].trim()
        } else {
            item.trim()
        };
        if looks_like_sql_aggregate_expr(col) {
            continue;
        }
        if !col.is_empty() && !is_literal_constant(col) {
            columns.push(col.to_string());
        }
    }

    columns
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    #[test]
    fn test_has_as_measure() {
        assert!(has_as_measure(
            "CREATE VIEW foo AS SELECT SUM(x) AS MEASURE revenue FROM t"
        ));
        assert!(!has_as_measure(
            "CREATE VIEW foo AS SELECT SUM(x) AS revenue FROM t"
        ));
    }

    #[test]
    fn test_has_aggregate_function() {
        assert!(has_aggregate_function("SELECT AGGREGATE(revenue) FROM foo"));
        assert!(has_aggregate_function("SELECT AGGREGATE (revenue) FROM foo"));
        assert!(has_aggregate_function("SELECT \"AGGREGATE\"(revenue) FROM foo"));
        assert!(has_aggregate_function("SELECT schema.AGGREGATE(revenue) FROM foo"));
        assert!(has_aggregate_function(
            "SELECT \"schema\".\"AGGREGATE\" (revenue) FROM foo"
        ));
        assert!(!has_aggregate_function("SELECT TOTAL_AGGREGATE(revenue) FROM foo"));
        assert!(!has_aggregate_function("SELECT \"TOTAL_AGGREGATE\"(revenue) FROM foo"));
        assert!(!has_aggregate_function("SELECT myaggregate(revenue) FROM foo"));
        assert!(!has_aggregate_function("SELECT SUM(amount) FROM foo"));
    }

    #[test]
    fn test_extract_dimension_columns_ignores_aggregate_with_space() {
        let cols = extract_dimension_columns_from_select(
            "SELECT region, AGGREGATE (revenue) FROM sales_v",
        );
        assert_eq!(cols, vec!["region".to_string()]);

        let cols = extract_dimension_columns_from_select(
            "SELECT region, AGGREGATE (revenue) AT (ALL region) FROM sales_v",
        );
        assert_eq!(cols, vec!["region".to_string()]);

        let cols = extract_dimension_columns_from_select(
            "SELECT AGGREGATE (revenue) FROM sales_v",
        );
        assert!(cols.is_empty());
    }

    #[test]
    fn test_extract_dimension_columns_excludes_standard_aggregates() {
        let cols = extract_dimension_columns_from_select(
            "SELECT region, COUNT(*) AS c, SUM(amount) AS s, ANY_VALUE(flag) AS f FROM sales GROUP BY ROLLUP(region)",
        );
        assert_eq!(cols, vec!["region".to_string()]);
    }

    #[test]
    fn test_extract_dimension_columns_keeps_non_aggregate_suffix() {
        let cols = extract_dimension_columns_from_select(
            "SELECT region, TOTAL_AGGREGATE(revenue) FROM sales_v",
        );
        assert_eq!(
            cols,
            vec!["region".to_string(), "TOTAL_AGGREGATE(revenue)".to_string()]
        );
    }

    #[test]
    fn test_extract_dimension_columns_ignores_quoted_and_qualified_aggregate() {
        let cols = extract_dimension_columns_from_select(
            "SELECT region, \"AGGREGATE\"(revenue) FROM sales_v",
        );
        assert_eq!(cols, vec!["region".to_string()]);

        let cols = extract_dimension_columns_from_select(
            "SELECT region, schema.AGGREGATE(revenue) FROM sales_v",
        );
        assert_eq!(cols, vec!["region".to_string()]);
    }

    #[test]
    fn test_is_literal_constant() {
        // Integers
        assert!(is_literal_constant("1000"));
        assert!(is_literal_constant("0"));
        assert!(is_literal_constant("-1"));
        assert!(is_literal_constant("+42"));

        // Floats
        assert!(is_literal_constant("3.14"));
        assert!(is_literal_constant("-0.5"));
        assert!(is_literal_constant("100.0"));

        // String literals
        assert!(is_literal_constant("'hello'"));
        assert!(is_literal_constant("'it''s a test'"));

        // NULL, TRUE, FALSE (case insensitive)
        assert!(is_literal_constant("NULL"));
        assert!(is_literal_constant("null"));
        assert!(is_literal_constant("TRUE"));
        assert!(is_literal_constant("true"));
        assert!(is_literal_constant("FALSE"));
        assert!(is_literal_constant("false"));

        // Date/time literals
        assert!(is_literal_constant("DATE '2023-01-01'"));
        assert!(is_literal_constant("TIMESTAMP '2023-01-01 00:00:00'"));
        assert!(is_literal_constant("INTERVAL '1 day'"));
        assert!(is_literal_constant("TIME '12:00:00'"));

        // Not constants
        assert!(!is_literal_constant("year"));
        assert!(!is_literal_constant("region"));
        assert!(!is_literal_constant("UPPER(region)"));
        assert!(!is_literal_constant("year + 1"));
        assert!(!is_literal_constant(""));
        assert!(!is_literal_constant("SUM(amount)"));
    }

    #[test]
    fn test_extract_dimension_columns_skips_literal_constants() {
        // Integer constant only
        let cols = extract_dimension_columns_from_select(
            "SELECT 1000, AGGREGATE(revenue) FROM sales_v",
        );
        assert!(cols.is_empty());

        // String literal only
        let cols = extract_dimension_columns_from_select(
            "SELECT 'hello', AGGREGATE(revenue) FROM sales_v",
        );
        assert!(cols.is_empty());

        // Mixed: dimension + constant
        let cols = extract_dimension_columns_from_select(
            "SELECT year, 1000, AGGREGATE(revenue) FROM sales_v",
        );
        assert_eq!(cols, vec!["year".to_string()]);

        // NULL constant
        let cols = extract_dimension_columns_from_select(
            "SELECT NULL, AGGREGATE(revenue) FROM sales_v",
        );
        assert!(cols.is_empty());

        // Float constant
        let cols = extract_dimension_columns_from_select(
            "SELECT 3.14, AGGREGATE(revenue) FROM sales_v",
        );
        assert!(cols.is_empty());

        // Negative integer
        let cols = extract_dimension_columns_from_select(
            "SELECT -1, AGGREGATE(revenue) FROM sales_v",
        );
        assert!(cols.is_empty());
    }

    #[test]
    #[serial]
    fn test_process_create_view_basic() {
        clear_measure_views();

        let sql =
            "CREATE VIEW orders_summary AS SELECT status, SUM(amount) AS MEASURE revenue FROM orders";
        let result = process_create_view(sql);

        assert!(result.is_measure_view);
        assert_eq!(result.view_name, Some("orders_summary".to_string()));
        assert_eq!(result.measures.len(), 1);
        assert_eq!(result.measures[0].column_name, "revenue");
        assert_eq!(result.measures[0].expression, "SUM(amount)");
        assert!(result.clean_sql.contains("AS revenue"));
        assert!(!result.clean_sql.contains("AS MEASURE"));
    }

    #[test]
    #[serial]
    fn test_process_create_view_case_expression() {
        clear_measure_views();

        let sql = "CREATE VIEW v AS SELECT year, CASE WHEN SUM(x) > 100 THEN 1 ELSE 0 END AS MEASURE flag FROM t GROUP BY year";
        let result = process_create_view(sql);

        eprintln!("Result: {result:?}");
        eprintln!("Measures: {:?}", result.measures);
        assert!(result.is_measure_view);
        assert_eq!(result.measures.len(), 1);
        assert_eq!(result.measures[0].column_name, "flag");
        assert_eq!(
            result.measures[0].expression,
            "CASE WHEN SUM(x) > 100 THEN 1 ELSE 0 END"
        );

        // Now test that AGGREGATE(flag) works
        let query_sql = "SELECT year, AGGREGATE(flag) FROM v GROUP BY year";
        let expand_result = std::panic::catch_unwind(|| expand_aggregate(query_sql))
            .unwrap_or_else(|_| expand_aggregate_with_at(query_sql));
        eprintln!("Expand result: {expand_result:?}");
        // This should have expanded AGGREGATE(flag) to something
        assert!(expand_result.had_aggregate);
    }

    #[test]
    #[serial]
    fn test_process_create_view_derived_measure() {
        clear_measure_views();

        // Create view with base measures and a derived measure
        let sql = "CREATE VIEW financials_v AS SELECT year, SUM(revenue) AS MEASURE revenue, SUM(cost) AS MEASURE cost, revenue - cost AS MEASURE profit FROM financials GROUP BY year";
        let result = process_create_view(sql);

        eprintln!("Clean SQL: {}", result.clean_sql);
        eprintln!("Measures: {:?}", result.measures);

        assert!(result.is_measure_view);
        assert_eq!(result.measures.len(), 3);

        // Check base measures
        assert_eq!(result.measures[0].column_name, "revenue");
        assert_eq!(result.measures[0].expression, "SUM(revenue)");
        assert_eq!(result.measures[1].column_name, "cost");
        assert_eq!(result.measures[1].expression, "SUM(cost)");

        // Check derived measure
        assert_eq!(result.measures[2].column_name, "profit");
        assert_eq!(result.measures[2].expression, "revenue - cost");

        // Clean SQL should NOT contain the derived measure column
        assert!(result.clean_sql.contains("AS revenue"));
        assert!(result.clean_sql.contains("AS cost"));
        assert!(!result.clean_sql.contains("AS profit"));
        assert!(!result.clean_sql.contains("revenue - cost"));
    }

    #[test]
    #[serial]
    fn test_process_create_view_without_group_by() {
        clear_measure_views();

        // Per the paper, views can define measures without GROUP BY
        let sql = r#"CREATE VIEW orders_extended AS
SELECT
    id,
    product,
    region,
    amount,
    SUM(amount) AS MEASURE revenue
FROM orders"#;
        let result = process_create_view(sql);

        eprintln!("is_measure_view: {}", result.is_measure_view);
        eprintln!("clean_sql: {}", result.clean_sql);
        eprintln!("measures: {:?}", result.measures);

        assert!(result.is_measure_view);
        assert_eq!(result.measures.len(), 1);
        assert_eq!(result.measures[0].column_name, "revenue");
        // The clean_sql should NOT contain the measure column at all for no-groupby views
        // because it can't be evaluated without a GROUP BY
    }

    #[test]
    fn test_has_at_syntax() {
        assert!(has_at_syntax(
            "SELECT AGGREGATE(revenue) AT (ALL status) FROM orders"
        ));
        assert!(!has_at_syntax("SELECT AGGREGATE(revenue) FROM orders"));
    }

    #[test]
    fn test_parse_at_modifier_all() {
        let result = parse_at_modifier("ALL status").unwrap();
        assert_eq!(result, ContextModifier::All("status".to_string()));
    }

    #[test]
    fn test_parse_at_modifier_set() {
        let result = parse_at_modifier("SET month = month - 1").unwrap();
        assert_eq!(
            result,
            ContextModifier::Set("month".to_string(), "month - 1".to_string())
        );
    }

    #[test]
    fn test_parse_at_modifier_where() {
        let result = parse_at_modifier("WHERE region = 'US'").unwrap();
        assert_eq!(result, ContextModifier::Where("region = 'US'".to_string()));
    }

    #[test]
    fn test_parse_at_modifier_where_strips_qualifier() {
        let result = parse_at_modifier("WHERE sales.region = 'US'").unwrap();
        assert_eq!(result, ContextModifier::Where("region = 'US'".to_string()));
    }

    #[test]
    fn test_extract_aggregate_with_at() {
        let sql = "SELECT status, AGGREGATE(revenue) AT (ALL status) FROM orders";
        let results = extract_aggregate_with_at(sql);

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "revenue");
        assert_eq!(results[0].1, ContextModifier::All("status".to_string()));
    }

    #[test]
    fn test_expand_at_to_sql_all() {
        // With no group_by_cols, AT (ALL dim) acts like grand total
        let expanded = expand_at_to_sql(
            "revenue",
            "SUM",
            &ContextModifier::All("status".to_string()),
            "orders",
            None,
            None,
            &[],
        );
        assert_eq!(expanded, "(SELECT SUM(revenue) FROM orders)");

        // With group_by_cols, AT (ALL dim) generates correlated subquery
        let expanded2 = expand_at_to_sql(
            "revenue",
            "SUM",
            &ContextModifier::All("region".to_string()),
            "sales_v",
            Some("_outer"),
            None,
            &["year".to_string(), "region".to_string()],
        );
        assert_eq!(
            expanded2,
            "(SELECT SUM(revenue) FROM sales_v _inner WHERE _inner.year IS NOT DISTINCT FROM _outer.year)"
        );
    }

    #[test]
    fn test_expand_at_to_sql_where() {
        let expanded = expand_at_to_sql(
            "revenue",
            "SUM",
            &ContextModifier::Where("orders.region = 'US'".to_string()),
            "orders",
            None,
            None,
            &[],
        );
        assert_eq!(
            expanded,
            "(SELECT SUM(revenue) FROM orders WHERE region = 'US')"
        );
    }

    #[test]
    fn test_expand_at_to_sql_set() {
        // With outer alias for proper correlation
        let expanded = expand_at_to_sql(
            "revenue",
            "SUM",
            &ContextModifier::Set("year".to_string(), "year - 1".to_string()),
            "sales_yearly",
            Some("_outer"),
            None,
            &[],
        );
        assert_eq!(
            expanded,
            "(SELECT SUM(revenue) FROM sales_yearly _inner WHERE _inner.year IS NOT DISTINCT FROM _outer.year - 1)"
        );
    }

    #[test]
    fn test_qualify_outer_reference() {
        assert_eq!(
            qualify_outer_reference("year - 1", "sales", "year"),
            "sales.year - 1"
        );
        assert_eq!(
            qualify_outer_reference("year + month", "t", "year"),
            "t.year + month"
        );
    }

    #[test]
    fn test_set_expression_requires_grouped_dimension() {
        let sql =
            "SELECT region, AGGREGATE(revenue) AT (SET year = year - 1) FROM sales_v";
        let result = expand_aggregate_with_at(sql);
        assert!(result.error.is_some());
        let message = result.error.unwrap();
        assert!(message.contains("SET year = year - 1"));
        assert!(message.contains("GROUP BY"));
    }

    #[test]
    fn test_set_expression_allows_grouped_dimension() {
        let sql =
            "SELECT year, region, AGGREGATE(revenue) AT (SET year = year - 1) FROM sales_v";
        let result = expand_aggregate_with_at(sql);
        assert!(result.error.is_none());
    }

    #[test]
    fn test_set_constant_allows_ungrouped_dimension() {
        let sql =
            "SELECT region, AGGREGATE(revenue) AT (SET year = 2023) FROM sales_v";
        let result = expand_aggregate_with_at(sql);
        assert!(result.error.is_none());
    }

    #[test]
    fn test_set_string_literal_allows_ungrouped_dimension() {
        let sql =
            "SELECT year, AGGREGATE(revenue) AT (SET region = 'region') FROM sales_v";
        let result = expand_aggregate_with_at(sql);
        assert!(result.error.is_none());
    }

    #[test]
    fn test_set_expression_requires_matching_grouping_alias() {
        let sql = "SELECT c.year, AGGREGATE(revenue) AT (SET year = year - 1) \
                   FROM orders_v o JOIN calendar_v c ON o.year = c.year \
                   GROUP BY c.year";
        let result = expand_aggregate_with_at(sql);
        assert!(result.error.is_some());
        let message = result.error.unwrap();
        assert!(message.contains("SET year = year - 1"));
    }

    #[test]
    fn test_set_expression_allows_matching_grouping_alias() {
        let sql = "SELECT o.year, AGGREGATE(revenue) AT (SET year = year - 1) FROM sales_v o GROUP BY o.year";
        let result = expand_aggregate_with_at(sql);
        assert!(result.error.is_none());
    }

    #[test]
    fn test_extract_agg_function() {
        assert_eq!(extract_agg_function("SUM(amount)"), "SUM");
        assert_eq!(extract_agg_function("count(*)"), "COUNT");
        assert_eq!(extract_agg_function("AVG(price)"), "AVG");
    }

    #[test]
    fn test_extract_group_by_columns() {
        let sql = "SELECT status, AGGREGATE(revenue) FROM orders_summary GROUP BY status";
        let cols = extract_group_by_columns(sql);
        assert_eq!(cols, vec!["status".to_string()]);

        let sql2 = "SELECT status, region FROM t GROUP BY status, region ORDER BY status";
        let cols2 = extract_group_by_columns(sql2);
        assert_eq!(cols2, vec!["status".to_string(), "region".to_string()]);
    }

    #[test]
    fn test_expand_curly_braces() {
        assert_eq!(
            expand_curly_braces("SELECT {revenue} FROM t"),
            "SELECT AGGREGATE(revenue) FROM t"
        );
    }

    #[test]
    #[ignore = "requires C++ parser FFI"]
    #[serial]
    fn test_rewrite_measure_at_refs() {
        clear_measure_views();
        store_measure_view(
            "sales_v",
            vec![ViewMeasure {
                column_name: "revenue".to_string(),
                expression: "SUM(amount)".to_string(),
                is_decomposable: true,
            }],
            "SELECT year, SUM(amount) AS revenue FROM sales GROUP BY year",
            Some("sales".to_string()),
        );

        let sql = "SELECT revenue AT (VISIBLE) FROM sales_v";
        let rewritten = rewrite_measure_at_refs(sql);
        assert_eq!(rewritten, "SELECT AGGREGATE(revenue) AT (VISIBLE) FROM sales_v");
        assert!(has_measure_at_refs(sql));
    }

    #[test]
    #[ignore = "requires C++ parser FFI"]
    #[serial]
    fn test_rewrite_implicit_measure_refs_uses_default_context() {
        clear_measure_views();
        store_measure_view(
            "sales_v",
            vec![ViewMeasure {
                column_name: "revenue".to_string(),
                expression: "SUM(amount)".to_string(),
                is_decomposable: true,
            }],
            "SELECT year, SUM(amount) AS revenue FROM sales GROUP BY year",
            Some("sales".to_string()),
        );

        let sql = "SELECT year, revenue FROM sales_v GROUP BY year";
        let rewritten = rewrite_implicit_measure_refs(sql);
        assert!(rewritten.contains("AGGREGATE(revenue) /*YARDSTICK_DEFAULT*/"));
    }

    #[test]
    fn test_rewrite_implicit_measure_refs_fallback_with_at_syntax() {
        let known_measures = HashSet::from([normalize_identifier_name("sumRevenue")]);
        let sql = "SELECT o.prodName, COUNT(*) AS c, o.sumRevenue AT (VISIBLE) AS rViz, o.sumRevenue AS r FROM paper_orders_v o GROUP BY ROLLUP(o.prodName)";
        let rewritten = rewrite_implicit_measure_refs_fallback(sql, &known_measures);

        assert!(rewritten.contains("o.sumRevenue AT (VISIBLE)"));
        assert!(rewritten.contains("AGGREGATE(o.sumRevenue) /*YARDSTICK_DEFAULT*/ AS r"));
    }

    #[test]
    fn test_has_implicit_measure_refs_fallback_ignores_count() {
        let known_measures = HashSet::from([normalize_identifier_name("sumRevenue")]);
        let sql = "SELECT o.prodName, COUNT(*) AS c, o.sumRevenue AT (VISIBLE) AS rViz, o.sumRevenue AS r FROM paper_orders_v o GROUP BY ROLLUP(o.prodName)";
        assert!(has_implicit_measure_refs_fallback(sql, &known_measures));

        let sql_no_plain_measure = "SELECT o.prodName, COUNT(*) AS c, o.sumRevenue AT (VISIBLE) AS rViz FROM paper_orders_v o GROUP BY ROLLUP(o.prodName)";
        assert!(!has_implicit_measure_refs_fallback(
            sql_no_plain_measure,
            &known_measures
        ));
    }

    #[test]
    fn test_extract_view_name() {
        assert_eq!(
            extract_view_name("CREATE VIEW foo AS SELECT 1"),
            Some("foo".to_string())
        );
        assert_eq!(
            extract_view_name("CREATE OR REPLACE VIEW bar AS SELECT 1"),
            Some("bar".to_string())
        );
    }

    #[test]
    #[serial]
    fn test_drop_measure_view_from_sql() {
        clear_measure_views();
        store_measure_view(
            "orders_v",
            vec![ViewMeasure {
                column_name: "revenue".to_string(),
                expression: "SUM(amount)".to_string(),
                is_decomposable: true,
            }],
            "SELECT year, SUM(amount) AS revenue FROM orders GROUP BY year",
            Some("orders".to_string()),
        );

        assert!(get_measure_view("orders_v").is_some());
        assert!(!drop_measure_view_from_sql(
            "DROP VIEW orders_v /* invalid tail */ extra"
        ));
        assert!(get_measure_view("orders_v").is_some());
        assert!(drop_measure_view_from_sql("DROP VIEW orders_v;"));
        assert!(get_measure_view("orders_v").is_none());
    }

    #[test]
    fn test_extract_drop_view_name_variants() {
        assert_eq!(
            extract_drop_view_name("DROP VIEW IF EXISTS analytics.orders_v"),
            Some("orders_v".to_string())
        );
        assert_eq!(
            extract_drop_view_name("drop view if exists \"Analytics\".\"Order.View\""),
            Some("Order.View".to_string())
        );
        assert_eq!(
            extract_drop_view_name("DROP VIEW /* keep */ IF EXISTS [dbo].[Orders View]"),
            Some("Orders View".to_string())
        );
        assert_eq!(extract_drop_view_name("DROP VIEW orders_v extra"), None);
    }

    #[test]
    fn test_extract_base_relation_sql_with_cte() {
        let view_query = "WITH base AS (SELECT * FROM orders) \
                          SELECT year, COUNT(*) AS cnt FROM base GROUP BY year";
        let base_relation = extract_base_relation_sql(view_query).unwrap();
        assert_eq!(
            base_relation,
            "WITH base AS (SELECT * FROM orders) SELECT * FROM base"
        );
    }

    #[test]
    fn test_extract_table_name_from_sql() {
        assert_eq!(
            extract_table_name_from_sql("SELECT * FROM orders"),
            Some("orders".to_string())
        );
        assert_eq!(
            extract_table_name_from_sql("SELECT a, b FROM sales_summary GROUP BY a"),
            Some("sales_summary".to_string())
        );
    }

    #[test]
    #[serial]
    fn test_expand_aggregate_with_at_set() {
        // Setup: register a measure view
        clear_measure_views();
        store_measure_view(
            "sales_yearly",
            vec![ViewMeasure {
                column_name: "revenue".to_string(),
                expression: "SUM(amount)".to_string(),
                is_decomposable: true,
            }],
            "SELECT year, SUM(amount) AS revenue FROM sales GROUP BY year",
            Some("sales".to_string()),
        );

        // Test AT (SET) expansion
        let sql = "SELECT year, AGGREGATE(revenue), AGGREGATE(revenue) AT (SET year = year - 1) AS prev_year FROM sales_yearly";
        let result = expand_aggregate_with_at(sql);

        eprintln!("Expanded SQL: {}", result.expanded_sql);
        assert!(result.had_aggregate);
        assert!(result.expanded_sql.contains("_outer"));
        assert!(result.expanded_sql.contains("_inner"));
    }

    #[test]
    #[serial]
    fn test_expand_aggregate_no_group_by() {
        // Test that queries without GROUP BY get implicit GROUP BY with dimension columns
        clear_measure_views();
        store_measure_view(
            "sales_v",
            vec![ViewMeasure {
                column_name: "revenue".to_string(),
                expression: "SUM(amount)".to_string(),
                is_decomposable: true,
            }],
            "SELECT year, region, SUM(amount) AS revenue FROM sales GROUP BY ALL",
            Some("sales".to_string()),
        );

        let sql = r#"SELECT year, region, AGGREGATE(revenue) AS revenue, AGGREGATE(revenue) AT (ALL region) AS year_total, AGGREGATE(revenue) / AGGREGATE(revenue) AT (ALL region) AS pct FROM sales_v"#;
        let result = expand_aggregate_with_at(sql);

        eprintln!("Expanded SQL: {}", result.expanded_sql);
        assert!(result.had_aggregate);
        // Should have GROUP BY with explicit columns (not GROUP BY ALL)
        let upper = result.expanded_sql.to_uppercase();
        assert!(upper.contains("GROUP BY"));
        assert!(upper.contains("YEAR"));
        assert!(upper.contains("REGION"));
        // Should correlate on year (not region, since AT ALL region removes it)
        assert!(result.expanded_sql.contains("_inner.year IS NOT DISTINCT FROM _outer.year"));
    }

    #[test]
    fn test_parse_current_in_expr() {
        assert_eq!(
            parse_current_in_expr("CURRENT year - 1"),
            "CURRENT year - 1"
        );
        assert_eq!(
            parse_current_in_expr("CURRENT year + CURRENT month"),
            "CURRENT year + CURRENT month"
        );
        assert_eq!(parse_current_in_expr("year - 1"), "year - 1");
    }

    #[test]
    fn test_parse_at_modifier_with_current() {
        let result = parse_at_modifier("SET year = CURRENT year - 1").unwrap();
        assert_eq!(
            result,
            ContextModifier::Set("year".to_string(), "CURRENT year - 1".to_string())
        );
    }

    #[test]
    fn test_resolve_current_in_expr_uses_null_when_unconstrained() {
        let resolved = resolve_current_in_expr(
            "CURRENT year - 1",
            &[String::from("region")],
            None,
            None,
        );
        assert_eq!(resolved, "NULL - 1");
    }

    #[test]
    fn test_resolve_current_in_expr_uses_dimension_when_grouped() {
        let resolved = resolve_current_in_expr(
            "CURRENT year - 1",
            &[String::from("year")],
            None,
            None,
        );
        assert_eq!(resolved, "year - 1");
    }

    #[test]
    fn test_resolve_current_in_expr_uses_dimension_when_where_constrained() {
        let resolved = resolve_current_in_expr(
            "CURRENT year - 1",
            &[],
            Some("year = 2024"),
            None,
        );
        assert_eq!(resolved, "year - 1");
    }

    #[test]
    fn test_chained_at_modifiers() {
        // Multiple ALL modifiers should combine to AllGlobal
        let sql = "SELECT AGGREGATE(revenue) AT (ALL year) AT (ALL region) FROM t";
        let results = extract_aggregate_with_at(sql);

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].1, ContextModifier::AllGlobal);
    }

    #[test]
    fn test_parse_at_modifier_all_global() {
        // The parser expects closing paren to distinguish AT (ALL) from AT (ALL dim)
        // Test via extract_aggregate_with_at instead
        let sql = "SELECT AGGREGATE(revenue) AT (ALL) FROM t";
        let results = extract_aggregate_with_at(sql);

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].1, ContextModifier::AllGlobal);
    }

    #[test]
    fn test_parse_at_modifier_visible() {
        let result = parse_at_modifier("VISIBLE").unwrap();
        assert_eq!(result, ContextModifier::Visible);
    }

    #[test]
    fn test_expand_at_to_sql_all_global() {
        let expanded = expand_at_to_sql(
            "revenue",
            "SUM",
            &ContextModifier::AllGlobal,
            "orders",
            None,
            None,
            &[],
        );
        assert_eq!(expanded, "(SELECT SUM(revenue) FROM orders)");
    }

    #[test]
    fn test_expand_at_to_sql_visible() {
        let expanded = expand_at_to_sql(
            "revenue",
            "SUM",
            &ContextModifier::Visible,
            "orders",
            None,
            None,
            &[],
        );
        // VISIBLE without outer_where returns simple aggregate
        assert_eq!(expanded, "SUM(revenue)");
    }

    #[test]
    fn test_expand_at_to_sql_visible_with_outer_where() {
        let expanded = expand_at_to_sql(
            "revenue",
            "SUM",
            &ContextModifier::Visible,
            "orders",
            None,
            Some("region = 'US'"),
            &[],
        );
        // VISIBLE with outer_where applies the WHERE clause
        assert_eq!(
            expanded,
            "(SELECT SUM(revenue) FROM orders WHERE region = 'US')"
        );
    }

    #[test]
    fn test_extract_where_clause() {
        let sql = "SELECT x FROM t WHERE region = 'US' GROUP BY x";
        assert_eq!(extract_where_clause(sql), Some("region = 'US'".to_string()));

        let sql2 = "SELECT x FROM t GROUP BY x";
        assert_eq!(extract_where_clause(sql2), None);

        let sql3 = "SELECT x FROM t WHERE a = 1 AND b = 2 ORDER BY x";
        assert_eq!(
            extract_where_clause(sql3),
            Some("a = 1 AND b = 2".to_string())
        );
    }

    #[test]
    fn test_expand_modifiers_to_sql_chained_all() {
        // Chaining AT (ALL region) AT (ALL category) should correlate on remaining dims (year)
        let modifiers = vec![
            ContextModifier::All("region".to_string()),
            ContextModifier::All("category".to_string()),
        ];
        let group_by = vec![
            "year".to_string(),
            "region".to_string(),
            "category".to_string(),
        ];
        let expanded = expand_modifiers_to_sql(
            "revenue",
            "SUM",
            &modifiers,
            "sales_v",
            Some("_outer"),
            None,
            &group_by,
        );
        // Should correlate on year only (not grand total)
        assert_eq!(
            expanded,
            "(SELECT SUM(revenue) FROM sales_v _inner WHERE _inner.year IS NOT DISTINCT FROM _outer.year)"
        );
    }

    #[test]
    fn test_expand_modifiers_all_dims_removed() {
        // When all GROUP BY dimensions are removed, should produce grand total
        let modifiers = vec![
            ContextModifier::All("year".to_string()),
            ContextModifier::All("region".to_string()),
        ];
        let group_by = vec!["year".to_string(), "region".to_string()];
        let expanded = expand_modifiers_to_sql(
            "revenue", "SUM", &modifiers, "sales_v", None, None, &group_by,
        );
        assert_eq!(expanded, "(SELECT SUM(revenue) FROM sales_v)");
    }

    #[test]
    fn test_expand_modifiers_set_then_all() {
        // SET year = year - 1, then ALL year should produce grand total (SET is overridden)
        let modifiers = vec![
            ContextModifier::Set("year".to_string(), "year - 1".to_string()),
            ContextModifier::All("year".to_string()),
        ];
        let group_by = vec!["year".to_string()];
        let expanded = expand_modifiers_to_sql(
            "revenue",
            "SUM",
            &modifiers,
            "sales_v",
            Some("_outer"),
            None,
            &group_by,
        );
        // ALL year should override SET year, resulting in grand total
        assert_eq!(expanded, "(SELECT SUM(revenue) FROM sales_v)");
    }

    #[test]
    fn test_expand_derived_measure_expr() {
        // Create a measure view with revenue and cost measures
        let mv = MeasureView {
            view_name: "sales_v".to_string(),
            measures: vec![
                ViewMeasure {
                    column_name: "revenue".to_string(),
                    expression: "SUM(amount)".to_string(),
                    is_decomposable: true,
                },
                ViewMeasure {
                    column_name: "cost".to_string(),
                    expression: "SUM(expense)".to_string(),
                    is_decomposable: true,
                },
            ],
            base_query: "".to_string(),
            base_table: Some("sales".to_string()),
            base_relation_sql: None,
            dimension_exprs: HashMap::new(),
            group_by_cols: Vec::new(),
        };

        // Simple subtraction
        let expanded = expand_derived_measure_expr("revenue - cost", &mv);
        assert_eq!(expanded, "SUM(revenue) - SUM(cost)");

        // With parentheses
        let expanded2 = expand_derived_measure_expr("(revenue - cost) / revenue", &mv);
        assert_eq!(expanded2, "(SUM(revenue) - SUM(cost)) / SUM(revenue)");

        // Non-measure identifiers preserved
        let expanded3 = expand_derived_measure_expr("revenue * 100", &mv);
        assert_eq!(expanded3, "SUM(revenue) * 100");
    }

    #[test]
    fn test_expand_derived_measure_expr_ignores_string_literals() {
        let mv = MeasureView {
            view_name: "sales_v".to_string(),
            measures: vec![ViewMeasure {
                column_name: "revenue".to_string(),
                expression: "SUM(amount)".to_string(),
                is_decomposable: true,
            }],
            base_query: "".to_string(),
            base_table: Some("sales".to_string()),
            base_relation_sql: None,
            dimension_exprs: HashMap::new(),
            group_by_cols: Vec::new(),
        };

        let expanded =
            expand_derived_measure_expr("CASE WHEN status = 'revenue' THEN revenue ELSE 0 END", &mv);
        assert_eq!(
            expanded,
            "CASE WHEN status = 'revenue' THEN SUM(revenue) ELSE 0 END"
        );
    }

    #[test]
    fn test_extract_table_and_alias() {
        // No alias
        assert_eq!(
            extract_table_and_alias_from_sql("SELECT * FROM orders"),
            Some(("orders".to_string(), None))
        );

        // With alias (no AS)
        assert_eq!(
            extract_table_and_alias_from_sql("SELECT * FROM orders o"),
            Some(("orders".to_string(), Some("o".to_string())))
        );

        // With AS keyword
        assert_eq!(
            extract_table_and_alias_from_sql("SELECT * FROM sales_v AS s"),
            Some(("sales_v".to_string(), Some("s".to_string())))
        );

        // With WHERE clause after
        assert_eq!(
            extract_table_and_alias_from_sql("SELECT * FROM orders o WHERE x = 1"),
            Some(("orders".to_string(), Some("o".to_string())))
        );

        // With GROUP BY after (no alias)
        assert_eq!(
            extract_table_and_alias_from_sql("SELECT x FROM orders GROUP BY x"),
            Some(("orders".to_string(), None))
        );

        // Quoted qualified names with dots inside identifiers
        assert_eq!(
            extract_table_and_alias_from_sql(
                "SELECT * FROM \"sales.schema\".\"orders.table\" o WHERE 1=1"
            ),
            Some(("orders.table".to_string(), Some("o".to_string())))
        );

        // Alias after comments (no AS)
        assert_eq!(
            extract_table_and_alias_from_sql("SELECT * FROM orders /* source */ o"),
            Some(("orders".to_string(), Some("o".to_string())))
        );

        // Alias after comments (with AS)
        assert_eq!(
            extract_table_and_alias_from_sql("SELECT * FROM orders AS /* source */ o"),
            Some(("orders".to_string(), Some("o".to_string())))
        );
    }

    #[test]
    fn test_insert_primary_table_alias_skips_comments() {
        let sql = "SELECT year, AGGREGATE(revenue) AT (SET year = year - 1) FROM sales_v /* c */ GROUP BY year";
        let updated = insert_primary_table_alias(sql, "_outer").unwrap();
        assert!(updated.contains("FROM sales_v _outer /* c */ GROUP BY"));

        let sql_with_alias = "SELECT year FROM sales_v /* c */ s GROUP BY year";
        assert!(insert_primary_table_alias(sql_with_alias, "_outer").is_none());
    }

    #[test]
    #[serial]
    fn test_expand_with_user_alias() {
        // Setup: register a measure view
        clear_measure_views();
        store_measure_view(
            "sales_v",
            vec![ViewMeasure {
                column_name: "revenue".to_string(),
                expression: "SUM(amount)".to_string(),
                is_decomposable: true,
            }],
            "SELECT year, SUM(amount) AS revenue FROM sales GROUP BY year",
            Some("sales".to_string()),
        );

        // Test with user-provided alias "s"
        let sql = "SELECT s.year, AGGREGATE(revenue) AT (SET year = year - 1) FROM sales_v s GROUP BY s.year";
        let result = expand_aggregate_with_at(sql);

        eprintln!("Expanded SQL: {}", result.expanded_sql);
        assert!(result.had_aggregate);
        // Should use "s" not "_outer"
        assert!(result.expanded_sql.contains("s.year"));
        assert!(!result.expanded_sql.contains("_outer"));
    }

    #[test]
    fn test_extract_alias_multiline() {
        // Test with newline before alias (as in actual SQL test)
        let sql = "SELECT s.year
FROM sales_yearly s
GROUP BY s.year";
        let result = extract_table_and_alias_from_sql(sql);
        eprintln!("Result: {result:?}");
        assert_eq!(
            result,
            Some(("sales_yearly".to_string(), Some("s".to_string())))
        );
    }

    // =========================================================================
    // JOIN Support Tests (using FFI parser)
    // =========================================================================

    #[test]
    #[ignore = "requires C++ library to be linked"]
    fn test_extract_from_clause_info_simple() {
        let sql = "SELECT * FROM orders";
        let info = extract_from_clause_info_ffi(sql);
        assert_eq!(info.tables.len(), 1);
        assert!(info.tables.contains_key("orders"));
        assert_eq!(info.primary_table.as_ref().unwrap().name, "orders");
        assert!(!info.primary_table.as_ref().unwrap().has_alias);
    }

    #[test]
    #[ignore = "requires C++ library to be linked"]
    fn test_extract_from_clause_info_with_alias() {
        let sql = "SELECT * FROM orders o";
        let info = extract_from_clause_info_ffi(sql);
        assert_eq!(info.tables.len(), 1);
        assert!(info.tables.contains_key("o"));
        let pt = info.primary_table.as_ref().unwrap();
        assert_eq!(pt.name, "orders");
        assert_eq!(pt.effective_name, "o");
        assert!(pt.has_alias);
    }

    #[test]
    #[ignore = "requires C++ library to be linked"]
    fn test_extract_from_clause_info_join() {
        let sql = "SELECT * FROM orders o JOIN customers c ON o.customer_id = c.id";
        let info = extract_from_clause_info_ffi(sql);
        eprintln!("Tables: {:?}", info.tables);
        assert_eq!(info.tables.len(), 2);
        assert!(info.tables.contains_key("o"));
        assert!(info.tables.contains_key("c"));
        assert_eq!(info.tables.get("o").unwrap().name, "orders");
        assert_eq!(info.tables.get("c").unwrap().name, "customers");
    }

    #[test]
    #[ignore = "requires C++ library to be linked"]
    fn test_extract_from_clause_info_multiple_joins() {
        let sql = "SELECT * FROM orders o \
                   JOIN customers c ON o.customer_id = c.id \
                   LEFT JOIN products p ON o.product_id = p.id";
        let info = extract_from_clause_info_ffi(sql);
        assert_eq!(info.tables.len(), 3);
        assert!(info.tables.contains_key("o"));
        assert!(info.tables.contains_key("c"));
        assert!(info.tables.contains_key("p"));
    }

    #[test]
    #[serial]
    fn test_resolve_measure_source() {
        clear_measure_views();
        store_measure_view(
            "orders_v",
            vec![ViewMeasure {
                column_name: "revenue".to_string(),
                expression: "SUM(amount)".to_string(),
                is_decomposable: true,
            }],
            "SELECT year, SUM(amount) AS revenue FROM orders GROUP BY year",
            Some("orders".to_string()),
        );

        // Should find revenue in orders_v
        let resolved = resolve_measure_source("revenue", "fallback");
        assert_eq!(resolved.agg_fn, "SUM");
        assert_eq!(resolved.source_view, "orders_v");
        assert!(resolved.derived_expr.is_none()); // Not a derived measure
        assert!(resolved.is_decomposable);
        assert_eq!(resolved.base_table, Some("orders".to_string()));

        // Unknown measure should fallback
        let resolved2 = resolve_measure_source("unknown", "fallback");
        assert_eq!(resolved2.agg_fn, "SUM");
        assert_eq!(resolved2.source_view, "fallback");
        assert!(resolved2.derived_expr.is_none());
        assert!(resolved2.is_decomposable); // Fallback assumes decomposable
    }

    #[test]
    #[ignore = "requires C++ library to be linked"]
    fn test_find_alias_for_view() {
        let sql = "SELECT * FROM orders_v o JOIN customers c ON o.id = c.order_id";
        let info = extract_from_clause_info_ffi(sql);

        // orders_v has alias "o"
        assert_eq!(find_alias_for_view(&info, "orders_v"), Some("o"));

        // customers has alias "c"
        assert_eq!(find_alias_for_view(&info, "customers"), Some("c"));

        // nonexistent table
        assert_eq!(find_alias_for_view(&info, "nonexistent"), None);
    }

    #[test]
    #[serial]
    fn test_expand_aggregate_with_join_measure_from_first_table() {
        clear_measure_views();
        store_measure_view(
            "orders_v",
            vec![ViewMeasure {
                column_name: "revenue".to_string(),
                expression: "SUM(amount)".to_string(),
                is_decomposable: true,
            }],
            "SELECT year, SUM(amount) AS revenue FROM orders GROUP BY year",
            Some("orders".to_string()),
        );

        // Query with JOIN - measure should expand using orders_v
        let sql = "SELECT o.year, c.name, AGGREGATE(revenue) AT (ALL) \
                   FROM orders_v o JOIN customers c ON o.customer_id = c.id";
        let result = expand_aggregate_with_at(sql);

        eprintln!("Expanded SQL: {}", result.expanded_sql);
        assert!(result.had_aggregate);
        // The subquery should be FROM orders_v (the measure's source view)
        assert!(result.expanded_sql.contains("FROM orders_v"));
    }

    #[test]
    #[serial]
    fn test_expand_aggregate_with_join_measure_from_second_table() {
        clear_measure_views();
        store_measure_view(
            "orders_v",
            vec![ViewMeasure {
                column_name: "revenue".to_string(),
                expression: "SUM(amount)".to_string(),
                is_decomposable: true,
            }],
            "SELECT year, SUM(amount) AS revenue FROM orders GROUP BY year",
            Some("orders".to_string()),
        );

        // Query with measure's source table as SECOND table in JOIN
        // This is the key test - previously would have used "customers" as table
        let sql = "SELECT c.name, AGGREGATE(revenue) AT (ALL) \
                   FROM customers c JOIN orders_v o ON c.order_id = o.id";
        let result = expand_aggregate_with_at(sql);

        eprintln!("Expanded SQL: {}", result.expanded_sql);
        assert!(result.had_aggregate);
        // Should still use orders_v (the measure's source), not customers
        assert!(result.expanded_sql.contains("FROM orders_v"));
    }

    #[test]
    fn test_parse_at_modifier_ad_hoc_dimension() {
        // Ad hoc dimension with function expression
        let result = parse_at_modifier("SET MONTH(order_date) = 2").unwrap();
        assert_eq!(
            result,
            ContextModifier::Set("MONTH(order_date)".to_string(), "2".to_string())
        );

        // ALL with function expression
        let result = parse_at_modifier("ALL YEAR(created_at)").unwrap();
        assert_eq!(result, ContextModifier::All("YEAR(created_at)".to_string()));

        // Nested function
        let result = parse_at_modifier("SET EXTRACT(month FROM date) = 6").unwrap();
        assert_eq!(
            result,
            ContextModifier::Set("EXTRACT(month FROM date)".to_string(), "6".to_string())
        );
    }

    #[test]
    fn test_qualify_where_for_inner_with_functions() {
        // Function calls should not get _inner. prefix
        assert_eq!(
            qualify_where_for_inner("MONTH(order_date) = 2"),
            "MONTH(_inner.order_date) = 2"
        );

        // Multiple columns in function
        assert_eq!(
            qualify_where_for_inner("DATEDIFF(start_date, end_date) > 30"),
            "DATEDIFF(_inner.start_date, _inner.end_date) > 30"
        );

        // Mix of functions and plain columns
        assert_eq!(
            qualify_where_for_inner("year = 2023 AND MONTH(date) = 6"),
            "_inner.year = 2023 AND MONTH(_inner.date) = 6"
        );

        // DATE literal keyword should not get _inner. prefix
        assert_eq!(
            qualify_where_for_inner("year between date '2023-01-01' and date '2025-01-01'"),
            "_inner.year between date '2023-01-01' and date '2025-01-01'"
        );
        assert_eq!(
            qualify_where_for_inner("event_time >= time '10:30:00'"),
            "_inner.event_time >= time '10:30:00'"
        );
        assert_eq!(
            qualify_where_for_inner("created_at < timestamp '2025-01-01 00:00:00'"),
            "_inner.created_at < timestamp '2025-01-01 00:00:00'"
        );
        assert_eq!(
            qualify_where_for_inner("created_at < timestamptz '2025-01-01 00:00:00+00'"),
            "_inner.created_at < timestamptz '2025-01-01 00:00:00+00'"
        );
        assert_eq!(
            qualify_where_for_inner("duration > interval '1 day'"),
            "_inner.duration > interval '1 day'"
        );
    }

    #[test]
    fn test_qualify_where_for_outer_basic() {
        assert_eq!(
            qualify_where_for_outer("region = 'US'", "o"),
            "o.region = 'US'"
        );
        assert_eq!(
            qualify_where_for_outer("sales.region = 'US'", "o"),
            "sales.region = 'US'"
        );
        assert_eq!(
            qualify_where_for_outer("cast(order_date as date) = date '2023-01-01'", "o"),
            "cast(o.order_date as date) = date '2023-01-01'"
        );
    }

    // =========================================================================
    // COUNT(DISTINCT) / Non-Decomposable Measure Tests
    // =========================================================================

    #[test]
    fn test_is_count_distinct() {
        // Should detect COUNT(DISTINCT ...)
        assert!(is_count_distinct("COUNT(DISTINCT user_id)"));
        assert!(is_count_distinct("COUNT( DISTINCT customer_id)"));
        assert!(is_count_distinct("count(distinct name)"));

        // Should NOT detect regular COUNT or other aggregates
        assert!(!is_count_distinct("COUNT(user_id)"));
        assert!(!is_count_distinct("COUNT(*)"));
        assert!(!is_count_distinct("SUM(amount)"));
        assert!(!is_count_distinct("AVG(price)"));
    }

    #[test]
    fn test_is_non_decomposable() {
        assert!(is_non_decomposable("COUNT(DISTINCT user_id)"));
        assert!(is_non_decomposable("median(value)"));
        assert!(is_non_decomposable("percentile_cont(0.5) within group (order by value)"));
        assert!(is_non_decomposable("quantile_cont(value, 0.5)"));
        assert!(is_non_decomposable("quantile_disc(value, 0.5)"));
        assert!(is_non_decomposable("mode(value)"));

        assert!(!is_non_decomposable("SUM(value)"));
        assert!(!is_non_decomposable("AVG(value)"));
    }

    #[test]
    fn test_rewrite_percentile_within_group() {
        let sql = "SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY value) AS p50";
        let rewritten = rewrite_percentile_within_group(sql);
        assert!(rewritten.contains("QUANTILE_CONT(value, 0.5)"));
    }

    #[test]
    #[serial]
    fn test_process_create_view_count_distinct() {
        clear_measure_views();

        let sql = "CREATE VIEW orders_v AS \
                   SELECT year, region, COUNT(DISTINCT customer_id) AS MEASURE unique_customers \
                   FROM orders";
        let result = process_create_view(sql);

        eprintln!("Result: {result:?}");
        eprintln!("Clean SQL: {}", result.clean_sql);

        assert!(result.is_measure_view);
        assert_eq!(result.measures.len(), 1);
        assert_eq!(result.measures[0].column_name, "unique_customers");
        assert_eq!(
            result.measures[0].expression,
            "COUNT(DISTINCT customer_id)"
        );

        // Verify the view gets the measure registered
        let mv = get_measure_view("orders_v").unwrap();
        assert!(!mv.measures[0].is_decomposable); // COUNT DISTINCT is non-decomposable
        assert_eq!(mv.base_table, Some("orders".to_string()));

        // The clean SQL should keep the COUNT(DISTINCT) column for base queries
        assert!(result.clean_sql.contains("COUNT(DISTINCT"));
        assert!(result.clean_sql.contains("unique_customers"));
        assert!(!result.clean_sql.contains("AS MEASURE"));
    }


    #[test]
    #[serial]
    fn test_expand_count_distinct_plain_aggregate() {
        clear_measure_views();

        // Register a view with COUNT(DISTINCT) measure
        store_measure_view(
            "orders_v",
            vec![ViewMeasure {
                column_name: "unique_customers".to_string(),
                expression: "COUNT(DISTINCT customer_id)".to_string(),
                is_decomposable: false,
            }],
            "SELECT year, region FROM orders GROUP BY year, region",
            Some("orders".to_string()),
        );

        // Plain AGGREGATE should fast-path to the view's measure column
        let sql = "SELECT year, region, AGGREGATE(unique_customers) FROM orders_v GROUP BY year, region";
        let result = expand_aggregate_with_at(sql);

        eprintln!("Expanded SQL: {}", result.expanded_sql);
        assert!(result.had_aggregate);
        assert!(result.error.is_none());
        assert!(result.expanded_sql.contains("MAX("));
        assert!(result.expanded_sql.contains("unique_customers"));
        assert!(!result.expanded_sql.contains("COUNT(DISTINCT"));
    }

    #[test]
    #[serial]
    fn test_expand_count_distinct_at_where() {
        clear_measure_views();

        store_measure_view(
            "orders_v",
            vec![ViewMeasure {
                column_name: "unique_customers".to_string(),
                expression: "COUNT(DISTINCT customer_id)".to_string(),
                is_decomposable: false,
            }],
            "SELECT year, region FROM orders GROUP BY year, region",
            Some("orders".to_string()),
        );

        // AT (WHERE) should work with COUNT(DISTINCT)
        let sql = "SELECT year, AGGREGATE(unique_customers) AT (WHERE region = 'US') \
                   FROM orders_v GROUP BY year";
        let result = expand_aggregate_with_at(sql);

        eprintln!("Expanded SQL: {}", result.expanded_sql);
        assert!(result.had_aggregate);
        assert!(result.error.is_none());
        // Should contain the WHERE condition in the subquery
        assert!(
            result.expanded_sql.contains("region = 'US'")
                || result.expanded_sql.contains("_inner.region = 'US'")
        );
    }

    #[test]
    #[serial]
    fn test_expand_count_distinct_at_all() {
        clear_measure_views();

        store_measure_view(
            "orders_v",
            vec![ViewMeasure {
                column_name: "unique_customers".to_string(),
                expression: "COUNT(DISTINCT customer_id)".to_string(),
                is_decomposable: false,
            }],
            "SELECT year, region FROM orders GROUP BY year, region",
            Some("orders".to_string()),
        );

        // AT (ALL region) should recompute COUNT(DISTINCT) with remaining context
        let sql = "SELECT year, AGGREGATE(unique_customers) AT (ALL region) \
                   FROM orders_v GROUP BY year";
        let result = expand_aggregate_with_at(sql);

        eprintln!("Expanded SQL: {}", result.expanded_sql);
        assert!(result.had_aggregate);
        assert!(result.error.is_none());
        assert!(result.expanded_sql.contains("COUNT(DISTINCT customer_id)"));
        assert!(result.expanded_sql.contains("LEFT JOIN"));
        assert!(result.expanded_sql.contains("IS NOT DISTINCT FROM _outer.year"));
    }

    #[test]
    #[serial]
    fn test_expand_count_distinct_at_all_global() {
        clear_measure_views();

        store_measure_view(
            "orders_v",
            vec![ViewMeasure {
                column_name: "unique_customers".to_string(),
                expression: "COUNT(DISTINCT customer_id)".to_string(),
                is_decomposable: false,
            }],
            "SELECT year, region FROM orders GROUP BY year, region",
            Some("orders".to_string()),
        );

        // AT (ALL) - grand total should recompute COUNT(DISTINCT)
        let sql = "SELECT year, AGGREGATE(unique_customers) AT (ALL) \
                   FROM orders_v GROUP BY year";
        let result = expand_aggregate_with_at(sql);

        assert!(result.had_aggregate);
        assert!(result.error.is_none());
        assert!(result.expanded_sql.contains("COUNT(DISTINCT customer_id)"));
        assert!(result.expanded_sql.contains("FROM (SELECT * FROM orders"));
        assert!(!result.expanded_sql.contains("_inner."));
    }

    #[test]
    #[serial]
    fn test_expand_count_distinct_at_set() {
        clear_measure_views();

        store_measure_view(
            "orders_v",
            vec![ViewMeasure {
                column_name: "unique_customers".to_string(),
                expression: "COUNT(DISTINCT customer_id)".to_string(),
                is_decomposable: false,
            }],
            "SELECT year, region FROM orders GROUP BY year, region",
            Some("orders".to_string()),
        );

        // AT (SET year = year - 1) should recompute COUNT(DISTINCT)
        let sql = "SELECT year, AGGREGATE(unique_customers) AT (SET year = year - 1) \
                   FROM orders_v GROUP BY year";
        let result = expand_aggregate_with_at(sql);

        assert!(result.had_aggregate);
        assert!(result.error.is_none());
        assert!(result.expanded_sql.contains("LEFT JOIN"));
        assert!(result.expanded_sql.contains("year - 1"));
    }

    #[test]
    #[serial]
    fn test_expand_count_distinct_uses_base_relation_join_where() {
        clear_measure_views();

        let mut views = MEASURE_VIEWS.lock().unwrap();
        views.insert(
            "orders_v".to_string(),
            MeasureView {
                view_name: "orders_v".to_string(),
                measures: vec![ViewMeasure {
                    column_name: "unique_customers".to_string(),
                    expression: "COUNT(DISTINCT customer_id)".to_string(),
                    is_decomposable: false,
                }],
                base_query: "SELECT year, region FROM orders".to_string(),
                base_table: None,
                base_relation_sql: Some(
                    "SELECT * FROM orders o JOIN regions r ON o.region_id = r.id WHERE o.status = 'paid'"
                        .to_string(),
                ),
                dimension_exprs: HashMap::new(),
                group_by_cols: Vec::new(),
            },
        );
        drop(views);

        let sql =
            "SELECT year, AGGREGATE(unique_customers) FROM orders_v GROUP BY year";
        let result = expand_aggregate_with_at(sql);

        assert!(result.had_aggregate);
        assert!(result.error.is_none());
        assert!(result.expanded_sql.contains("LEFT JOIN (SELECT"));
        assert!(result.expanded_sql.contains("JOIN regions r ON o.region_id = r.id"));
        assert!(result.expanded_sql.contains("o.status = 'paid'"));
    }

    #[test]
    #[serial]
    fn test_expand_count_distinct_alias_dimension_expr() {
        clear_measure_views();

        let mut dimension_exprs = HashMap::new();
        dimension_exprs.insert(
            "year".to_string(),
            "date_trunc('year', order_date)".to_string(),
        );

        let mut views = MEASURE_VIEWS.lock().unwrap();
        views.insert(
            "orders_v".to_string(),
            MeasureView {
                view_name: "orders_v".to_string(),
                measures: vec![ViewMeasure {
                    column_name: "unique_customers".to_string(),
                    expression: "COUNT(DISTINCT customer_id)".to_string(),
                    is_decomposable: false,
                }],
                base_query: "SELECT year FROM orders".to_string(),
                base_table: None,
                base_relation_sql: Some("SELECT * FROM orders".to_string()),
                dimension_exprs,
                group_by_cols: Vec::new(),
            },
        );
        drop(views);

        let sql = "SELECT year, region, AGGREGATE(unique_customers) AT (ALL region) \
                   FROM orders_v GROUP BY year, region";
        let result = expand_aggregate_with_at(sql);

        assert!(result.had_aggregate);
        assert!(result.error.is_none());
        assert!(result.expanded_sql.contains("LEFT JOIN"));
        assert!(result
            .expanded_sql
            .contains("date_trunc('year', _inner.order_date) AS dim_0"));
        assert!(result.expanded_sql.contains("IS NOT DISTINCT FROM _outer.year"));
    }

    #[test]
    #[serial]
    fn test_expand_count_distinct_set_op_base_relation() {
        clear_measure_views();

        store_measure_view(
            "orders_v",
            vec![ViewMeasure {
                column_name: "unique_customers".to_string(),
                expression: "COUNT(DISTINCT customer_id)".to_string(),
                is_decomposable: false,
            }],
            "SELECT year, region FROM a UNION ALL SELECT year, region FROM b",
            None,
        );

        let sql =
            "SELECT year, AGGREGATE(unique_customers) FROM orders_v GROUP BY year";
        let result = expand_aggregate_with_at(sql);

        assert!(result.had_aggregate);
        assert!(result.error.is_none());
        assert!(result.expanded_sql.contains("LEFT JOIN"));
        assert!(result.expanded_sql.contains(
            "FROM (SELECT * FROM (SELECT year, region FROM a UNION ALL SELECT year, region FROM b)) _inner"
        ));
    }
}
