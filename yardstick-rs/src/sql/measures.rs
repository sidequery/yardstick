//! Julian Hyde's "Measures in SQL" support
//!
//! Implements:
//! - `AS MEASURE` in CREATE VIEW (arXiv:2406.00251)
//! - `AGGREGATE()` function expansion
//! - `AT (ALL dim)`, `AT (SET dim = expr)` context modifiers
//!
//! Reference: https://arxiv.org/abs/2406.00251

use std::collections::HashMap;
use std::sync::Mutex;

use nom::{
    branch::alt,
    bytes::complete::{tag_no_case, take_until, take_while, take_while1},
    character::complete::{char, multispace0, multispace1},
    combinator::{opt, recognize},
    sequence::{delimited, pair, tuple},
    IResult,
};
use once_cell::sync::Lazy;
use sqlparser::ast::{
    Expr, Function, FunctionArg, FunctionArgExpr, FunctionArguments, Query, Select, SelectItem,
    SetExpr, Statement, TableFactor,
};
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::Parser;

use crate::error::{Result, YardstickError};

// =============================================================================
// Data Structures
// =============================================================================

/// Stored measure definition from CREATE VIEW ... AS MEASURE
#[derive(Debug, Clone)]
pub struct ViewMeasure {
    pub column_name: String,
    pub expression: String,
}

/// Stored view with measures
#[derive(Debug, Clone)]
pub struct MeasureView {
    pub view_name: String,
    pub measures: Vec<ViewMeasure>,
    pub base_query: String,
}

/// Global storage for measure views (in-memory catalog)
static MEASURE_VIEWS: Lazy<Mutex<HashMap<String, MeasureView>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

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
        Ok((input, format!("{}({})", name, args)))
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
    let mut remaining = sql;
    while !remaining.is_empty() {
        if let Ok((rest, _)) =
            take_until::<_, _, nom::error::Error<&str>>("AGGREGATE(")(remaining)
        {
            // Verify it's actually AGGREGATE( not part of another word
            if rest.len() >= 10 {
                return true;
            }
        }
        if remaining.len() > 1 {
            remaining = &remaining[1..];
        } else {
            break;
        }
    }
    // Also check case-insensitive
    sql.to_uppercase().contains("AGGREGATE(")
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
                result.push_str(&format!("AGGREGATE({})", ident));
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
        Err(nom::Err::Error(nom::error::Error::new(input, nom::error::ErrorKind::Char)))
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
/// Returns the dimension name with CURRENT stripped
fn parse_current_in_expr(expr: &str) -> String {
    // Replace "CURRENT dim" with just "dim" (let outer reference handle it)
    let mut result = expr.to_string();
    let expr_upper = expr.to_uppercase();

    // Find all "CURRENT identifier" patterns
    let mut search_pos = 0;
    while let Some(pos) = expr_upper[search_pos..].find("CURRENT ") {
        let abs_pos = search_pos + pos;
        let after_current = abs_pos + 8; // "CURRENT " is 8 chars

        // Extract the identifier after CURRENT
        let remaining = &expr[after_current..];
        if let Ok((_, ident)) = identifier(remaining) {
            // Replace "CURRENT ident" with just "ident"
            let pattern = &expr[abs_pos..after_current + ident.len()];
            result = result.replacen(pattern, ident, 1);
        }
        search_pos = abs_pos + 1;
    }

    result
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
    Ok((
        input,
        ContextModifier::Set(dim, processed_expr),
    ))
}

/// Parse AT (WHERE condition)
fn at_where(input: &str) -> IResult<&str, ContextModifier> {
    let (input, _) = tag_no_case("WHERE")(input)?;
    let (input, _) = multispace1(input)?;
    // Take rest as condition (until closing paren, handled by caller)
    let (input, cond) = take_while(|c: char| c != ')')(input)?;
    Ok((input, ContextModifier::Where(cond.trim().to_string())))
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
        .map_err(|e| YardstickError::Validation(format!("Failed to parse AT modifier: {:?}", e)))
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
    let mut modifiers = Vec::new();
    let mut remaining = input.trim();

    // Keep parsing modifiers until we can't parse any more
    while !remaining.is_empty() {
        let result = at_modifier(remaining);
        match result {
            Ok((rest, modifier)) => {
                modifiers.push(modifier);
                remaining = rest.trim();
            }
            Err(_) => break,
        }
    }

    if modifiers.is_empty() {
        Err(nom::Err::Error(nom::error::Error::new(input, nom::error::ErrorKind::Tag)))
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
        let at_start: IResult<&str, _> = tuple((
            multispace0,
            tag_no_case("AT"),
            multispace0,
            char('('),
        ))(remaining);

        match at_start {
            Ok((after_open, _)) => {
                // Find the matching close paren
                if let Ok((after_content, content)) = balanced_parens(after_open) {
                    // Skip the closing paren
                    if let Ok((after_close, _)) = char::<_, nom::error::Error<&str>>(')')(after_content) {
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
pub fn extract_aggregate_with_at_full(sql: &str) -> Vec<(String, Vec<ContextModifier>, usize, usize)> {
    let mut results = Vec::new();
    let sql_upper = sql.to_uppercase();
    let mut search_pos = 0;

    while let Some(agg_offset) = sql_upper[search_pos..].find("AGGREGATE(") {
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
                let all_are_all = modifiers.iter().all(|m| matches!(m, ContextModifier::All(_) | ContextModifier::AllGlobal));
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

    while let Some(agg_offset) = sql_upper[search_pos..].find("AGGREGATE(") {
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

/// Extract table name from SQL FROM clause
pub fn extract_table_name_from_sql(sql: &str) -> Option<String> {
    extract_table_and_alias_from_sql(sql).map(|(name, _)| name)
}

/// Extract table name and optional alias from SQL FROM clause
/// Returns (table_name, Option<alias>)
pub fn extract_table_and_alias_from_sql(sql: &str) -> Option<(String, Option<String>)> {
    // Normalize whitespace to handle newlines/tabs in SQL
    let normalized: String = sql.chars().map(|c| if c.is_whitespace() { ' ' } else { c }).collect();
    let normalized_upper = normalized.to_uppercase();
    let from_pos = normalized_upper.find(" FROM ")?;
    let after_from = &normalized[from_pos..];

    // Parse: FROM table_name [AS] [alias]
    let (rest, _) = multispace1::<_, nom::error::Error<&str>>(after_from).ok()?;
    let (rest, _) = tag_no_case::<_, _, nom::error::Error<&str>>("FROM")(rest).ok()?;
    let (rest, _) = multispace1::<_, nom::error::Error<&str>>(rest).ok()?;
    let (rest, table) = identifier(rest).ok()?;

    // Check for alias (optional AS keyword followed by identifier)
    let rest_trimmed = rest.trim_start();

    // Check for end of FROM clause (WHERE, GROUP, ORDER, etc.)
    let rest_upper = rest_trimmed.to_uppercase();
    if rest_trimmed.is_empty()
        || rest_upper.starts_with("WHERE")
        || rest_upper.starts_with("GROUP")
        || rest_upper.starts_with("ORDER")
        || rest_upper.starts_with("LIMIT")
        || rest_upper.starts_with("HAVING")
        || rest_upper.starts_with("JOIN")
        || rest_trimmed.starts_with(';')
    {
        return Some((table.to_string(), None));
    }

    // Try to parse optional AS keyword
    let after_as = if rest_upper.starts_with("AS ") {
        &rest_trimmed[3..]
    } else {
        rest_trimmed
    };

    // Parse the alias identifier
    if let Ok((_, alias)) = identifier(after_as.trim_start()) {
        // Make sure alias isn't a keyword
        let alias_upper = alias.to_uppercase();
        if matches!(alias_upper.as_str(), "WHERE" | "GROUP" | "ORDER" | "LIMIT" | "HAVING" | "JOIN") {
            return Some((table.to_string(), None));
        }
        Some((table.to_string(), Some(alias.to_string())))
    } else {
        Some((table.to_string(), None))
    }
}

/// Extract WHERE clause from SQL query
pub fn extract_where_clause(sql: &str) -> Option<String> {
    // Normalize whitespace
    let normalized: String = sql.chars().map(|c| if c.is_whitespace() { ' ' } else { c }).collect();
    let normalized_upper = normalized.to_uppercase();

    // Find WHERE keyword (not inside a subquery)
    let mut depth = 0;
    let mut where_start = None;
    let chars: Vec<char> = normalized.chars().collect();

    for i in 0..chars.len() {
        match chars[i] {
            '(' => depth += 1,
            ')' => depth -= 1,
            _ => {
                if depth == 0 && i + 6 <= chars.len() {
                    let word = &normalized_upper[i..i+6];
                    if word == "WHERE " {
                        where_start = Some(i + 6);
                        break;
                    }
                }
            }
        }
    }

    let start = where_start?;

    // Find end of WHERE clause (GROUP BY, ORDER BY, LIMIT, HAVING, or end)
    let rest_upper = &normalized_upper[start..];
    let end_keywords = [" GROUP BY", " ORDER BY", " LIMIT ", " HAVING ", ";"];
    let mut end_pos = normalized.len();

    for kw in end_keywords {
        if let Some(pos) = rest_upper.find(kw) {
            if start + pos < end_pos {
                end_pos = start + pos;
            }
        }
    }

    let where_content = normalized[start..end_pos].trim().to_string();
    if where_content.is_empty() {
        None
    } else {
        Some(where_content)
    }
}

// =============================================================================
// Nom Parsers - Expression Helpers
// =============================================================================

/// Parse aggregation function name: SUM, COUNT, AVG, MIN, MAX, etc.
fn agg_function_name(input: &str) -> IResult<&str, &str> {
    alt((
        tag_no_case("SUM"),
        tag_no_case("COUNT"),
        tag_no_case("AVG"),
        tag_no_case("MIN"),
        tag_no_case("MAX"),
        tag_no_case("STDDEV"),
        tag_no_case("VARIANCE"),
    ))(input)
}

/// Extract aggregation function from expression like "SUM(amount)"
pub fn extract_agg_function(expr: &str) -> String {
    agg_function_name(expr.trim())
        .map(|(_, name)| name.to_uppercase())
        .unwrap_or_else(|_| "SUM".to_string())
}

/// Extract aggregation function name from full expression
/// "SUM(amount)" -> "sum"
pub fn extract_aggregation_function(expr: &str) -> Option<String> {
    let (_, (name, _)) = function_call(expr.trim()).ok()?;
    Some(name.to_lowercase())
}

/// Find any aggregation function inside an expression
/// "CASE WHEN SUM(x) > 100 THEN 1 ELSE 0 END" -> Some("sum")
fn find_aggregation_in_expression(expr: &str) -> Option<String> {
    let expr_upper = expr.to_uppercase();
    for agg in ["SUM", "COUNT", "AVG", "MIN", "MAX"] {
        if expr_upper.contains(&format!("{}(", agg)) {
            return Some(agg.to_lowercase());
        }
    }
    None
}

/// Expand derived measure expression by replacing measure references with their aggregations
/// "revenue - cost" with measures [revenue=SUM(amount), cost=SUM(expense)]
/// -> "SUM(revenue) - SUM(cost)"
fn expand_derived_measure_expr(expr: &str, measure_view: &MeasureView) -> String {
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

            // Check if this identifier is a measure name
            if let Some(m) = measure_view.measures.iter().find(|m| m.column_name.eq_ignore_ascii_case(&ident)) {
                // Get the aggregation function from this measure's expression
                if let Some(agg_fn) = extract_aggregation_function(&m.expression) {
                    // Replace measure name with AGG(measure_name)
                    result.push_str(&format!("{}({})", agg_fn.to_uppercase(), ident));
                } else {
                    // Fallback to SUM if no aggregation found
                    result.push_str(&format!("SUM({})", ident));
                }
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

/// Qualify column references in a WHERE clause for use inside _inner subquery
/// "region = 'US'" -> "_inner.region = 'US'"
/// "year > 2020 AND region = 'US'" -> "_inner.year > 2020 AND _inner.region = 'US'"
fn qualify_where_for_inner(where_clause: &str) -> String {
    let mut result = String::new();
    let mut chars = where_clause.chars().peekable();

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
            let already_qualified = chars.peek() == Some(&'.');
            let is_function = chars.peek() == Some(&'(');

            // SQL keywords that should not be prefixed
            let keywords = ["AND", "OR", "NOT", "IN", "IS", "NULL", "TRUE", "FALSE",
                           "LIKE", "BETWEEN", "EXISTS", "CASE", "WHEN", "THEN", "ELSE", "END"];
            let is_keyword = keywords.iter().any(|kw| kw.eq_ignore_ascii_case(&ident));

            if !already_qualified && !is_keyword && !is_function {
                result.push_str("_inner.");
            }
            result.push_str(&ident);
        } else if c == '\'' {
            // String literal - copy as-is until closing quote
            result.push(c);
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
        }
    }

    result
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
        Ok((clean_sql, measures, view_name)) => {
            if !measures.is_empty() {
                if let Some(ref vn) = view_name {
                    let measure_view = MeasureView {
                        view_name: vn.clone(),
                        measures: measures.clone(),
                        base_query: clean_sql.clone(),
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
fn extract_measures_from_sql(sql: &str) -> Result<(String, Vec<ViewMeasure>, Option<String>)> {
    let view_name = extract_view_name(sql);
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

    // Collect all measure names for derived measure detection
    let measure_names: Vec<&str> = measure_infos.iter().map(|m| m.name.as_str()).collect();

    // Check if an expression is derived (references other measures, no aggregation)
    let is_derived = |expr: &str| -> bool {
        if extract_aggregation_function(expr).is_some() {
            return false; // Has aggregation, not derived
        }
        // Check if expression references any measure name
        for name in &measure_names {
            // Simple word boundary check
            let expr_lower = expr.to_lowercase();
            let name_lower = name.to_lowercase();
            if expr_lower.contains(&name_lower) {
                // More precise check: ensure it's a word boundary
                for (i, _) in expr_lower.match_indices(&name_lower) {
                    let before_ok = i == 0 || !expr.chars().nth(i - 1).unwrap_or(' ').is_alphanumeric();
                    let after_ok = i + name_lower.len() >= expr.len()
                        || !expr.chars().nth(i + name_lower.len()).unwrap_or(' ').is_alphanumeric();
                    if before_ok && after_ok {
                        return true;
                    }
                }
            }
        }
        false
    };

    // Generate replacements based on whether measure is base or derived
    // (start, end, replacement) - for derived, we remove the whole column
    let mut replacements: Vec<(usize, usize, String)> = Vec::new();

    for info in &measure_infos {
        if is_derived(&info.expression) {
            // Derived measure: remove entire column including preceding comma
            let mut remove_start = info.expr_start;
            // Look for preceding comma
            let before = &sql[..remove_start];
            if let Some(comma_pos) = before.rfind(',') {
                // Check that only whitespace between comma and expr_start
                let between = &before[comma_pos + 1..];
                if between.trim().is_empty() {
                    remove_start = comma_pos;
                }
            }
            replacements.push((remove_start, info.name_end, String::new()));
        } else {
            // Base measure: replace "AS MEASURE name" with "AS name"
            let pattern_start = info.name_end - " AS MEASURE ".len() - info.name.len()
                - (sql[info.expr_start..info.name_end].len() - info.expression.len() - " AS MEASURE ".len() - info.name.len());
            // Actually, simpler: find " AS MEASURE " before name_end
            let chunk = &sql[info.expr_start..info.name_end];
            if let Some(am_pos) = chunk.to_uppercase().find(" AS MEASURE ") {
                let abs_start = info.expr_start + am_pos;
                replacements.push((abs_start, info.name_end, format!(" AS {}", info.name)));
            }
        }
    }

    // Build measures list
    let measures: Vec<ViewMeasure> = measure_infos
        .into_iter()
        .map(|m| ViewMeasure {
            column_name: m.name,
            expression: m.expression,
        })
        .collect();

    // Apply replacements in reverse order
    let mut clean_sql = sql.to_string();
    replacements.sort_by(|a, b| b.0.cmp(&a.0));
    for (start, end, replacement) in replacements {
        clean_sql = format!("{}{}{}", &clean_sql[..start], replacement, &clean_sql[end..]);
    }

    Ok((clean_sql, measures, view_name))
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
pub fn expand_aggregate(sql: &str) -> AggregateExpandResult {
    if !has_aggregate_function(sql) {
        return AggregateExpandResult {
            had_aggregate: false,
            expanded_sql: sql.to_string(),
            error: None,
        };
    }

    let dialect = GenericDialect {};
    let statements = match Parser::parse_sql(&dialect, sql) {
        Ok(s) => s,
        Err(e) => {
            return AggregateExpandResult {
                had_aggregate: false,
                expanded_sql: sql.to_string(),
                error: Some(format!("SQL parse error: {e}")),
            };
        }
    };

    if statements.is_empty() {
        return AggregateExpandResult {
            had_aggregate: false,
            expanded_sql: sql.to_string(),
            error: Some("Empty SQL".to_string()),
        };
    }

    match &statements[0] {
        Statement::Query(query) => match expand_aggregate_in_query(query) {
            Ok(expanded) => AggregateExpandResult {
                had_aggregate: true,
                expanded_sql: expanded.to_string(),
                error: None,
            },
            Err(e) => AggregateExpandResult {
                had_aggregate: false,
                expanded_sql: sql.to_string(),
                error: Some(e.to_string()),
            },
        },
        _ => AggregateExpandResult {
            had_aggregate: false,
            expanded_sql: sql.to_string(),
            error: None,
        },
    }
}

fn expand_aggregate_in_query(query: &Query) -> Result<Query> {
    let body = match &*query.body {
        SetExpr::Select(select) => {
            let expanded = expand_aggregate_in_select(select)?;
            SetExpr::Select(Box::new(expanded))
        }
        other => other.clone(),
    };

    Ok(Query {
        body: Box::new(body),
        ..query.clone()
    })
}

fn expand_aggregate_in_select(select: &Select) -> Result<Select> {
    let table_name = extract_table_name(select);
    let views = MEASURE_VIEWS.lock().unwrap();
    let measure_view = table_name.as_ref().and_then(|name| views.get(name));

    let mut dimension_columns: Vec<Expr> = Vec::new();
    let mut has_aggregate = false;

    for item in &select.projection {
        let (expr, is_agg) = match item {
            SelectItem::UnnamedExpr(expr) => (Some(expr.clone()), contains_aggregate(expr)),
            SelectItem::ExprWithAlias { expr, .. } => (Some(expr.clone()), contains_aggregate(expr)),
            _ => (None, false),
        };
        if is_agg {
            has_aggregate = true;
        } else if let Some(e) = expr {
            dimension_columns.push(e);
        }
    }

    let projection: Vec<SelectItem> = select
        .projection
        .iter()
        .map(|item| expand_aggregate_in_select_item(item, measure_view, Some(&views)))
        .collect::<Result<Vec<_>>>()?;

    let group_by = if has_aggregate
        && select.group_by == sqlparser::ast::GroupByExpr::Expressions(vec![], vec![])
    {
        // Add GROUP BY with dimensions, or GROUP BY () for scalar aggregation
        sqlparser::ast::GroupByExpr::Expressions(dimension_columns, vec![])
    } else {
        select.group_by.clone()
    };

    Ok(Select {
        projection,
        group_by,
        ..select.clone()
    })
}

fn contains_aggregate(expr: &Expr) -> bool {
    match expr {
        Expr::Function(func) => func.name.to_string().to_uppercase() == "AGGREGATE",
        Expr::BinaryOp { left, right, .. } => contains_aggregate(left) || contains_aggregate(right),
        Expr::Nested(inner) => contains_aggregate(inner),
        // Scalar subqueries shouldn't go in GROUP BY - they're self-contained
        Expr::Subquery(_) => true,
        _ => false,
    }
}

fn extract_table_name(select: &Select) -> Option<String> {
    select.from.first().and_then(|table| {
        if let sqlparser::ast::TableFactor::Table { name, .. } = &table.relation {
            name.0.first().map(|i| i.value.clone())
        } else {
            None
        }
    })
}

// =============================================================================
// AST-based FROM clause extraction (for JOIN support)
// =============================================================================

/// Extract table info from a TableFactor AST node
#[allow(dead_code)] // Used in tests; useful for future full JOIN support
fn extract_table_info_from_factor(factor: &TableFactor) -> Option<TableInfo> {
    match factor {
        TableFactor::Table { name, alias, .. } => {
            let table_name = name.0.first()?.value.clone();
            let (effective_name, has_alias) = match alias {
                Some(a) => (a.name.value.clone(), true),
                None => (table_name.clone(), false),
            };
            Some(TableInfo {
                name: table_name,
                effective_name,
                has_alias,
            })
        }
        TableFactor::Derived { alias, .. } => {
            // Derived tables (subqueries) - use alias as both name and effective_name
            alias.as_ref().map(|a| TableInfo {
                name: a.name.value.clone(),
                effective_name: a.name.value.clone(),
                has_alias: true,
            })
        }
        TableFactor::NestedJoin { table_with_joins, alias } => {
            // For nested joins, prefer the alias if present
            if let Some(a) = alias {
                Some(TableInfo {
                    name: a.name.value.clone(),
                    effective_name: a.name.value.clone(),
                    has_alias: true,
                })
            } else {
                // Recursively get primary table from nested join
                extract_table_info_from_factor(&table_with_joins.relation)
            }
        }
        _ => None,
    }
}

/// Extract all tables from a SELECT's FROM clause, including JOINs
#[allow(dead_code)] // Used in tests; useful for future full JOIN support
fn extract_from_clause_info(select: &Select) -> FromClauseInfo {
    let mut info = FromClauseInfo::default();

    for (i, table_with_joins) in select.from.iter().enumerate() {
        // Extract the main relation
        if let Some(table_info) = extract_table_info_from_factor(&table_with_joins.relation) {
            if i == 0 && info.primary_table.is_none() {
                info.primary_table = Some(table_info.clone());
            }
            info.tables.insert(table_info.effective_name.clone(), table_info);
        }

        // Extract all JOINed tables
        for join in &table_with_joins.joins {
            if let Some(table_info) = extract_table_info_from_factor(&join.relation) {
                info.tables.insert(table_info.effective_name.clone(), table_info);
            }
        }
    }

    info
}

/// Look up which view contains a measure and return (agg_fn, source_view_name, derived_expr)
/// derived_expr is Some if this is a derived measure (should use expanded expression instead of AGG(name))
/// Prioritizes default_table (the query's FROM table), then searches other views for JOINs
fn resolve_measure_source(measure_name: &str, default_table: &str) -> (String, String, Option<String>) {
    let views = MEASURE_VIEWS.lock().unwrap();

    // First, check if the measure exists in the default table (query's primary table)
    if let Some(v) = views.get(default_table) {
        if let Some(m) = v.measures.iter().find(|m| m.column_name.eq_ignore_ascii_case(measure_name)) {
            let agg_fn = extract_agg_function(&m.expression);
            // Check if this is a derived measure (no aggregation, references other measures)
            if extract_aggregation_function(&m.expression).is_none() {
                let expanded = expand_derived_measure_expr(&m.expression, v);
                if expanded != m.expression {
                    return (agg_fn, default_table.to_string(), Some(expanded));
                }
            }
            return (agg_fn, default_table.to_string(), None);
        }
    }

    // If not found in default table, search other views (for JOIN support)
    views.iter()
        .find_map(|(view_name, v)| {
            v.measures.iter()
                .find(|m| m.column_name.eq_ignore_ascii_case(measure_name))
                .map(|m| {
                    let agg_fn = extract_agg_function(&m.expression);
                    // Check if this is a derived measure
                    if extract_aggregation_function(&m.expression).is_none() {
                        let expanded = expand_derived_measure_expr(&m.expression, v);
                        if expanded != m.expression {
                            return (agg_fn, view_name.clone(), Some(expanded));
                        }
                    }
                    (agg_fn, view_name.clone(), None)
                })
        })
        .unwrap_or_else(|| ("SUM".to_string(), default_table.to_string(), None))
}

/// Find the alias used for a view in the FROM clause
/// Returns the effective_name (alias) if the view is in the FROM clause
fn find_alias_for_view<'a>(from_info: &'a FromClauseInfo, view_name: &str) -> Option<&'a str> {
    from_info.tables.values()
        .find(|t| t.name.eq_ignore_ascii_case(view_name))
        .map(|t| t.effective_name.as_str())
}

fn expand_aggregate_in_select_item(
    item: &SelectItem,
    measure_view: Option<&MeasureView>,
    all_views: Option<MeasureViewsRef>,
) -> Result<SelectItem> {
    match item {
        SelectItem::UnnamedExpr(expr) => {
            let expanded = expand_aggregate_in_expr(expr, measure_view, all_views)?;
            Ok(SelectItem::UnnamedExpr(expanded))
        }
        SelectItem::ExprWithAlias { expr, alias } => {
            let expanded = expand_aggregate_in_expr(expr, measure_view, all_views)?;
            Ok(SelectItem::ExprWithAlias {
                expr: expanded,
                alias: alias.clone(),
            })
        }
        other => Ok(other.clone()),
    }
}

/// All measure views available for lookup (passed to avoid re-locking)
type MeasureViewsRef<'a> = &'a HashMap<String, MeasureView>;

fn expand_aggregate_in_expr(
    expr: &Expr,
    measure_view: Option<&MeasureView>,
    all_views: Option<MeasureViewsRef>,
) -> Result<Expr> {
    match expr {
        Expr::Function(func) => {
            let func_name = func.name.to_string().to_uppercase();

            if func_name == "AGGREGATE" {
                let measure_name = extract_aggregate_arg(func)?;

                // First try the primary measure view
                let mut measure = measure_view
                    .and_then(|mv| mv.measures.iter().find(|m| m.column_name == measure_name));

                // For JOINs: if not found in primary view, search all registered views
                let mut fallback_view: Option<&MeasureView> = None;
                if measure.is_none() {
                    if let Some(views) = all_views {
                        for v in views.values() {
                            if let Some(m) = v.measures.iter().find(|m| m.column_name.eq_ignore_ascii_case(&measure_name)) {
                                measure = Some(m);
                                fallback_view = Some(v);
                                break;
                            }
                        }
                    }
                }
                let effective_view = fallback_view.or(measure_view);

                if let Some(m) = measure {
                    let dialect = GenericDialect {};

                    // Try to extract aggregation function (e.g., SUM, COUNT)
                    // If that fails, look for aggregation inside the expression (e.g., CASE WHEN SUM(x)...)
                    // Then use that aggregation on the measure column
                    let expr_sql = if let Some(agg_fn) = extract_aggregation_function(&m.expression) {
                        format!("SELECT {}({})", agg_fn, measure_name)
                    } else if let Some(agg_fn) = find_aggregation_in_expression(&m.expression) {
                        // CASE WHEN SUM(x) > 100... → use SUM(measure_column)
                        format!("SELECT {}({})", agg_fn, measure_name)
                    } else if let Some(mv) = effective_view {
                        // Check if this is a derived measure (references other measures)
                        let expanded = expand_derived_measure_expr(&m.expression, mv);
                        if expanded != m.expression {
                            // Successfully expanded derived measure
                            format!("SELECT {}", expanded)
                        } else {
                            // No aggregation found, default to SUM
                            format!("SELECT SUM({})", measure_name)
                        }
                    } else {
                        // No aggregation found, default to SUM
                        format!("SELECT SUM({})", measure_name)
                    };

                    let stmts = Parser::parse_sql(&dialect, &expr_sql)
                        .map_err(|e| YardstickError::SqlParse(e.to_string()))?;

                    if let Some(Statement::Query(q)) = stmts.into_iter().next() {
                        if let SetExpr::Select(s) = *q.body {
                            if let Some(SelectItem::UnnamedExpr(e)) =
                                s.projection.into_iter().next()
                            {
                                return Ok(e);
                            }
                        }
                    }
                }

                return Err(YardstickError::Validation(format!(
                    "Measure '{}' not found",
                    measure_name
                )));
            }

            Ok(Expr::Function(func.clone()))
        }
        Expr::BinaryOp { left, op, right } => {
            let left = expand_aggregate_in_expr(left, measure_view, all_views)?;
            let right = expand_aggregate_in_expr(right, measure_view, all_views)?;
            Ok(Expr::BinaryOp {
                left: Box::new(left),
                op: op.clone(),
                right: Box::new(right),
            })
        }
        Expr::Nested(inner) => {
            let inner = expand_aggregate_in_expr(inner, measure_view, all_views)?;
            Ok(Expr::Nested(Box::new(inner)))
        }
        other => Ok(other.clone()),
    }
}

fn extract_aggregate_arg(func: &Function) -> Result<String> {
    match &func.args {
        FunctionArguments::List(args) => {
            if args.args.is_empty() {
                return Err(YardstickError::Validation(
                    "AGGREGATE() requires a measure name".to_string(),
                ));
            }
            match &args.args[0] {
                FunctionArg::Unnamed(FunctionArgExpr::Expr(Expr::Identifier(ident))) => {
                    Ok(ident.value.clone())
                }
                FunctionArg::Unnamed(FunctionArgExpr::Expr(Expr::CompoundIdentifier(parts))) => {
                    Ok(parts.last().map(|i| i.value.clone()).unwrap_or_default())
                }
                _ => Err(YardstickError::Validation(
                    "AGGREGATE() argument must be a measure name".to_string(),
                )),
            }
        }
        _ => Err(YardstickError::Validation(
            "AGGREGATE() requires arguments".to_string(),
        )),
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
    let measure_expr = format!("{}({})", agg_fn, measure_col);

    match modifier {
        ContextModifier::AllGlobal => {
            // Grand total - no WHERE clause, aggregate over entire table
            format!("(SELECT {} FROM {})", measure_expr, table_name)
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
                        let col_name = col.split('.').last().unwrap_or(col);
                        col_name.to_lowercase() != dim_lower
                    }
                })
                .collect();

            if correlating_dims.is_empty() {
                // No other dimensions - same as AllGlobal
                format!("(SELECT {} FROM {})", measure_expr, table_name)
            } else {
                // Correlate on remaining dimensions
                let where_clauses: Vec<_> = correlating_dims
                    .iter()
                    .map(|col| {
                        let col_is_expr = col.contains('(');
                        if col_is_expr {
                            // For expression dimensions, qualify column refs inside
                            let inner_expr = qualify_where_for_inner(col);
                            format!("{} = {}", inner_expr, col)
                        } else {
                            // Extract just the column name for _inner reference
                            let col_name = col.split('.').last().unwrap_or(col);
                            format!("_inner.{} = {}.{}", col_name, outer_ref, col_name)
                        }
                    })
                    .collect();
                format!(
                    "(SELECT {} FROM {} _inner WHERE {})",
                    measure_expr, table_name, where_clauses.join(" AND ")
                )
            }
        }
        ContextModifier::Set(dim, expr) => {
            // Use outer_alias for the correlated reference, falling back to table_name
            let outer_ref = outer_alias.unwrap_or(table_name);
            let qualified_expr = qualify_outer_reference(expr, outer_ref, dim);
            // For ad hoc dimensions (expressions like MONTH(date)), qualify column refs with _inner
            // For simple columns, just prefix with _inner.
            let inner_dim = if dim.contains('(') {
                // Expression: qualify column refs inside it
                qualify_where_for_inner(dim)
            } else {
                format!("_inner.{}", dim)
            };

            // SET condition for the specified dimension
            let set_condition = format!("{} = {}", inner_dim, qualified_expr);

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
                        let col_name = col.split('.').last().unwrap_or(col);
                        col_name.to_lowercase() != dim_lower
                    }
                })
                .map(|col| {
                    let col_is_expr = col.contains('(');
                    if col_is_expr {
                        let inner_expr = qualify_where_for_inner(col);
                        format!("{} = {}", inner_expr, col)
                    } else {
                        let col_name = col.split('.').last().unwrap_or(col);
                        format!("_inner.{} = {}.{}", col_name, outer_ref, col_name)
                    }
                })
                .collect();

            // Combine: SET condition + correlation on other dims
            // NOTE: Do NOT include outer_where - SET bypasses outer WHERE per paper
            let mut all_conditions = vec![set_condition];
            all_conditions.extend(correlation_conditions);

            format!(
                "(SELECT {} FROM {} _inner WHERE {})",
                measure_expr, table_name, all_conditions.join(" AND ")
            )
        }
        ContextModifier::Where(condition) => {
            format!(
                "(SELECT {} FROM {} WHERE {})",
                measure_expr, table_name, condition
            )
        }
        ContextModifier::Visible => {
            // VISIBLE means include outer query's WHERE clause AND respect GROUP BY context
            let outer_ref = outer_alias.unwrap_or(table_name);
            if group_by_cols.is_empty() {
                // No GROUP BY - just apply WHERE
                match outer_where {
                    Some(w) => format!("(SELECT {} FROM {} WHERE {})", measure_expr, table_name, w),
                    None => measure_expr,
                }
            } else {
                // Correlate on GROUP BY columns and apply WHERE
                let where_clauses: Vec<_> = group_by_cols
                    .iter()
                    .map(|col| {
                        // Extract just the column name for _inner reference
                        let col_name = col.split('.').last().unwrap_or(col);
                        format!("_inner.{} = {}.{}", col_name, outer_ref, col_name)
                    })
                    .collect();
                let full_where = match outer_where {
                    Some(w) => format!("{} AND {}", where_clauses.join(" AND "), w),
                    None => where_clauses.join(" AND "),
                };
                format!(
                    "(SELECT {} FROM {} _inner WHERE {})",
                    measure_expr, table_name, full_where
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
        return expand_at_to_sql(measure_col, agg_fn, &ContextModifier::Visible, table_name, outer_alias, outer_where, group_by_cols);
    }

    if modifiers.len() == 1 {
        return expand_at_to_sql(measure_col, agg_fn, &modifiers[0], table_name, outer_alias, outer_where, group_by_cols);
    }

    // Check if all modifiers are ALL (dimension or global)
    let all_are_all = modifiers.iter().all(|m| matches!(m, ContextModifier::All(_) | ContextModifier::AllGlobal));
    if all_are_all {
        // Check for explicit AllGlobal - that means grand total
        if modifiers.iter().any(|m| matches!(m, ContextModifier::AllGlobal)) {
            return expand_at_to_sql(measure_col, agg_fn, &ContextModifier::AllGlobal, table_name, outer_alias, outer_where, group_by_cols);
        }

        // Accumulate all dimensions to remove
        let removed_dims: Vec<&str> = modifiers.iter()
            .filter_map(|m| match m {
                ContextModifier::All(dim) => Some(dim.as_str()),
                _ => None,
            })
            .collect();

        // Filter group_by_cols to get remaining dimensions
        let remaining_cols: Vec<&String> = group_by_cols.iter()
            .filter(|col| {
                let col_name = col.split('.').last().unwrap_or(col).to_lowercase();
                !removed_dims.iter().any(|d| d.to_lowercase() == col_name)
            })
            .collect();

        if remaining_cols.is_empty() {
            // All dimensions removed = grand total
            return expand_at_to_sql(measure_col, agg_fn, &ContextModifier::AllGlobal, table_name, outer_alias, outer_where, group_by_cols);
        }

        // Generate correlation on remaining dimensions only
        let outer_ref = outer_alias.unwrap_or(table_name);
        let measure_expr = format!("{}({})", agg_fn, measure_col);
        let where_clauses: Vec<_> = remaining_cols.iter()
            .map(|col| {
                let col_name = col.split('.').last().unwrap_or(col);
                format!("_inner.{} = {}.{}", col_name, outer_ref, col_name)
            })
            .collect();
        return format!(
            "(SELECT {} FROM {} _inner WHERE {})",
            measure_expr, table_name, where_clauses.join(" AND ")
        );
    }

    // Apply modifiers right-to-left
    // For now, collect the effects:
    // - VISIBLE adds outer WHERE (but SET bypasses it per paper)
    // - ALL removes all filters
    // - SET changes a dimension and bypasses outer WHERE
    // - WHERE adds a filter

    // Check if SET is present - per paper, SET bypasses outer WHERE
    let has_set = modifiers.iter().any(|m| matches!(m, ContextModifier::Set(_, _)));

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
                        // Qualify column references with _inner
                        effective_where = Some(qualify_where_for_inner(w));
                    }
                }
            }
            ContextModifier::Where(cond) => {
                if !has_all_global {
                    // Qualify column references with _inner
                    effective_where = Some(qualify_where_for_inner(cond));
                }
            }
            ContextModifier::Set(dim, expr) => {
                // Skip SET if ALL(dim) was already processed (dimension removed from context)
                let dim_lower = dim.to_lowercase();
                if !has_all_global && !removed_dims.contains(&dim_lower) {
                    let outer_ref = outer_alias.unwrap_or(table_name);
                    let qualified_expr = qualify_outer_reference(expr, outer_ref, dim);
                    // For ad hoc dimensions (expressions), qualify column refs inside
                    let inner_dim = if dim.contains('(') {
                        qualify_where_for_inner(dim)
                    } else {
                        format!("_inner.{}", dim)
                    };
                    set_conditions.push(format!("{} = {}", inner_dim, qualified_expr));
                }
            }
        }
    }

    // Build final SQL
    let measure_expr = format!("{}({})", agg_fn, measure_col);

    if has_all_global && set_conditions.is_empty() {
        // Pure grand total
        return format!("(SELECT {} FROM {})", measure_expr, table_name);
    }

    // Filter group_by_cols to exclude removed dimensions
    let remaining_cols: Vec<&String> = group_by_cols.iter()
        .filter(|col| {
            let col_lower = col.to_lowercase();
            let col_name = col.split('.').last().unwrap_or(col).to_lowercase();
            // Check both full expression match and simple column match
            !removed_dims.iter().any(|d| *d == col_lower || *d == col_name)
        })
        .collect();

    // Build correlation conditions for remaining dimensions
    let outer_ref = outer_alias.unwrap_or(table_name);
    let correlation_conditions: Vec<String> = remaining_cols.iter()
        .map(|col| {
            let col_is_expr = col.contains('(');
            if col_is_expr {
                // For expression dimensions, qualify column refs inside
                let inner_expr = qualify_where_for_inner(col);
                format!("{} = {}", inner_expr, col)
            } else {
                let col_name = col.split('.').last().unwrap_or(col);
                format!("_inner.{} = {}.{}", col_name, outer_ref, col_name)
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
        format!("(SELECT {} FROM {})", measure_expr, table_name)
    } else {
        format!(
            "(SELECT {} FROM {} _inner WHERE {})",
            measure_expr, table_name, all_conditions.join(" AND ")
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
        return format!("(SELECT {} FROM {})", derived_expr, table_name);
    }

    // Check for AllGlobal (grand total)
    if modifiers.iter().any(|m| matches!(m, ContextModifier::AllGlobal)) {
        return format!("(SELECT {} FROM {})", derived_expr, table_name);
    }

    // Check if all modifiers are ALL (dimension)
    let all_are_all = modifiers.iter().all(|m| matches!(m, ContextModifier::All(_)));
    if all_are_all {
        // Accumulate dimensions to remove
        let removed_dims: Vec<&str> = modifiers.iter()
            .filter_map(|m| match m {
                ContextModifier::All(dim) => Some(dim.as_str()),
                _ => None,
            })
            .collect();

        // Filter group_by_cols to get remaining dimensions
        let remaining_cols: Vec<&String> = group_by_cols.iter()
            .filter(|col| {
                let col_lower = col.to_lowercase();
                let col_name = col.split('.').last().unwrap_or(col).to_lowercase();
                // Check both full expression match and simple column match
                !removed_dims.iter().any(|d| d.to_lowercase() == col_lower || d.to_lowercase() == col_name)
            })
            .collect();

        if remaining_cols.is_empty() {
            // All dimensions removed = grand total
            return format!("(SELECT {} FROM {})", derived_expr, table_name);
        }

        // Generate correlation on remaining dimensions
        let outer_ref = outer_alias.unwrap_or(table_name);
        let where_clauses: Vec<_> = remaining_cols.iter()
            .map(|col| {
                let col_is_expr = col.contains('(');
                if col_is_expr {
                    // For expression dimensions, qualify column refs inside
                    let inner_expr = qualify_where_for_inner(col);
                    format!("{} = {}", inner_expr, col)
                } else {
                    let col_name = col.split('.').last().unwrap_or(col);
                    format!("_inner.{} = {}.{}", col_name, outer_ref, col_name)
                }
            })
            .collect();
        return format!(
            "(SELECT {} FROM {} _inner WHERE {})",
            derived_expr, table_name, where_clauses.join(" AND ")
        );
    }

    // For other modifiers (SET, WHERE, VISIBLE), build conditions
    // Check if SET is present - per paper, SET bypasses outer WHERE
    let has_set = modifiers.iter().any(|m| matches!(m, ContextModifier::Set(_, _)));

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
                        effective_where = Some(qualify_where_for_inner(w));
                    }
                }
            }
            ContextModifier::Where(cond) => {
                if !has_all_global {
                    effective_where = Some(qualify_where_for_inner(cond));
                }
            }
            ContextModifier::Set(_, _) => {
                // SET doesn't apply directly to derived measures in the same way
                // The derived expression already includes the aggregations
            }
        }
    }

    if has_all_global {
        return format!("(SELECT {} FROM {})", derived_expr, table_name);
    }

    // Filter remaining dimensions
    let remaining_cols: Vec<&String> = group_by_cols.iter()
        .filter(|col| {
            let col_lower = col.to_lowercase();
            let col_name = col.split('.').last().unwrap_or(col).to_lowercase();
            // Check both full expression match and simple column match
            !removed_dims.iter().any(|d| *d == col_lower || *d == col_name)
        })
        .collect();

    // Build conditions
    let outer_ref = outer_alias.unwrap_or(table_name);
    let mut all_conditions: Vec<String> = remaining_cols.iter()
        .map(|col| {
            let col_is_expr = col.contains('(');
            if col_is_expr {
                // For expression dimensions, qualify column refs inside
                let inner_expr = qualify_where_for_inner(col);
                format!("{} = {}", inner_expr, col)
            } else {
                let col_name = col.split('.').last().unwrap_or(col);
                format!("_inner.{} = {}.{}", col_name, outer_ref, col_name)
            }
        })
        .collect();

    if let Some(w) = effective_where {
        all_conditions.push(w);
    }

    if all_conditions.is_empty() {
        format!("(SELECT {} FROM {})", derived_expr, table_name)
    } else {
        format!(
            "(SELECT {} FROM {} _inner WHERE {})",
            derived_expr, table_name, all_conditions.join(" AND ")
        )
    }
}

/// Expand AGGREGATE() with AT modifiers in SQL
pub fn expand_aggregate_with_at(sql: &str) -> AggregateExpandResult {
    if !has_at_syntax(sql) {
        return expand_aggregate(sql);
    }

    let at_patterns = extract_aggregate_with_at_full(sql);
    if at_patterns.is_empty() {
        return expand_aggregate(sql);
    }

    // Extract table info using string-based approach (works with AGGREGATE syntax)
    // Note: sqlparser can't parse AGGREGATE() since it's not standard SQL
    let (primary_table_name, existing_alias) = extract_table_and_alias_from_sql(sql)
        .unwrap_or_else(|| ("t".to_string(), None));

    // Build FromClauseInfo from string-based extraction for now
    // TODO: For proper JOIN support, we'd need to extract all tables from the FROM clause
    let mut from_info = FromClauseInfo::default();
    let primary_table = TableInfo {
        name: primary_table_name.clone(),
        effective_name: existing_alias.clone().unwrap_or_else(|| primary_table_name.clone()),
        has_alias: existing_alias.is_some(),
    };
    from_info.tables.insert(primary_table.effective_name.clone(), primary_table.clone());
    from_info.primary_table = Some(primary_table);

    // Extract outer WHERE clause for VISIBLE semantics
    let outer_where = extract_where_clause(sql);
    let outer_where_ref = outer_where.as_deref();

    // Extract GROUP BY columns for AT (ALL dim) correlation
    let group_by_cols = extract_group_by_columns(sql);

    // Check if any AT modifier needs correlation (for alias handling)
    let needs_outer_alias = at_patterns.iter().any(|(_, modifiers, _, _)| {
        modifiers.iter().any(|m| {
            matches!(m, ContextModifier::Set(_, _)) ||
            matches!(m, ContextModifier::All(_)) ||
            matches!(m, ContextModifier::Visible)
        })
    });

    let mut result_sql = sql.to_string();

    // Handle alias for the primary table if needed for correlation
    let primary_alias: Option<String> = if needs_outer_alias {
        if let Some(ref pt) = from_info.primary_table {
            if pt.has_alias {
                Some(pt.effective_name.clone())
            } else {
                // No alias on primary table, add _outer
                let from_pattern = format!("FROM {}", pt.name);
                let from_replacement = format!("FROM {} _outer", pt.name);
                result_sql = result_sql.replace(&from_pattern, &from_replacement);
                Some("_outer".to_string())
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
        // Look up which view contains this measure (for JOIN support)
        let (agg_fn, source_view, derived_expr) = resolve_measure_source(&measure_name, &primary_table_name);

        // Find the alias for this measure's source view in the FROM clause
        // If the source view is the primary table and we added _outer, use _outer
        let outer_alias = if let Some(ref pt) = from_info.primary_table {
            if pt.name.eq_ignore_ascii_case(&source_view) {
                // Source view is the primary table, use primary_alias (which may be _outer)
                primary_alias.clone()
            } else {
                // Source view is not the primary table, look it up in from_info
                find_alias_for_view(&from_info, &source_view)
                    .map(|s| s.to_string())
                    .or_else(|| primary_alias.clone())
            }
        } else {
            primary_alias.clone()
        };
        let outer_alias_ref = outer_alias.as_deref();

        // For derived measures, use the expanded expression instead of measure_name
        let (effective_measure, effective_agg) = if let Some(ref expr) = derived_expr {
            // Derived measure: use expanded expression directly (no wrapping AGG)
            (expr.as_str(), "".to_string())
        } else {
            (measure_name.as_str(), agg_fn)
        };

        let expanded = if derived_expr.is_some() {
            // For derived measures, build the subquery with the expanded expression
            expand_modifiers_to_sql_derived(effective_measure, &modifiers, &source_view, outer_alias_ref, outer_where_ref, &group_by_cols)
        } else {
            expand_modifiers_to_sql(&measure_name, &effective_agg, &modifiers, &source_view, outer_alias_ref, outer_where_ref, &group_by_cols)
        };
        result_sql = format!("{}{}{}", &result_sql[..start], expanded, &result_sql[end..]);
    }

    // Also expand plain AGGREGATE() calls (without AT modifiers) using text replacement
    let mut plain_calls = extract_all_aggregate_calls(&result_sql);
    plain_calls.sort_by(|a, b| b.1.cmp(&a.1)); // Sort by position descending

    for (measure_name, start, end) in plain_calls {
        let (agg_fn, _source_view, derived_expr) = resolve_measure_source(&measure_name, &primary_table_name);

        // For derived measures, use the expanded expression; otherwise use AGG(measure_name)
        let expanded = if let Some(expr) = derived_expr {
            expr
        } else {
            format!("{}({})", agg_fn, measure_name)
        };
        result_sql = format!("{}{}{}", &result_sql[..start], expanded, &result_sql[end..]);
    }

    // Check if there are still any remaining AGGREGATE calls (shouldn't be, but just in case)
    if has_aggregate_function(&result_sql) {
        return expand_aggregate(&result_sql);
    }

    AggregateExpandResult {
        had_aggregate: true,
        expanded_sql: result_sql,
        error: None,
    }
}

// =============================================================================
// Catalog Functions
// =============================================================================

pub fn store_measure_view(view_name: &str, measures: Vec<ViewMeasure>, base_query: &str) {
    let measure_view = MeasureView {
        view_name: view_name.to_string(),
        measures,
        base_query: base_query.to_string(),
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
    let sql_upper = sql.to_uppercase();
    let mut columns = Vec::new();

    if let Some(group_by_pos) = sql_upper.find("GROUP BY") {
        let after_group_by = &sql[group_by_pos + 8..];

        let end_pos = ["ORDER BY", "LIMIT", "HAVING", ";"]
            .iter()
            .filter_map(|kw| after_group_by.to_uppercase().find(kw))
            .min()
            .unwrap_or(after_group_by.len());

        let group_by_content = after_group_by[..end_pos].trim();

        for part in group_by_content.split(',') {
            let col = part.trim();
            if !col.is_empty() && !col.chars().all(|c| c.is_ascii_digit()) {
                columns.push(col.to_string());
            }
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
        assert!(!has_aggregate_function("SELECT SUM(amount) FROM foo"));
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

        eprintln!("Result: {:?}", result);
        eprintln!("Measures: {:?}", result.measures);
        assert!(result.is_measure_view);
        assert_eq!(result.measures.len(), 1);
        assert_eq!(result.measures[0].column_name, "flag");
        assert_eq!(result.measures[0].expression, "CASE WHEN SUM(x) > 100 THEN 1 ELSE 0 END");

        // Now test that AGGREGATE(flag) works
        let query_sql = "SELECT year, AGGREGATE(flag) FROM v GROUP BY year";
        let expand_result = expand_aggregate(query_sql);
        eprintln!("Expand result: {:?}", expand_result);
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
        assert_eq!(
            result,
            ContextModifier::Where("region = 'US'".to_string())
        );
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
            "(SELECT SUM(revenue) FROM sales_v _inner WHERE _inner.year = _outer.year)"
        );
    }

    #[test]
    fn test_expand_at_to_sql_where() {
        let expanded = expand_at_to_sql(
            "revenue",
            "SUM",
            &ContextModifier::Where("region = 'US'".to_string()),
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
            "(SELECT SUM(revenue) FROM sales_yearly _inner WHERE _inner.year = _outer.year - 1)"
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
            }],
            "SELECT year, SUM(amount) AS revenue FROM sales GROUP BY year",
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
    fn test_parse_current_in_expr() {
        // CURRENT year - 1 should become year - 1
        assert_eq!(parse_current_in_expr("CURRENT year - 1"), "year - 1");

        // Multiple CURRENT references
        assert_eq!(
            parse_current_in_expr("CURRENT year + CURRENT month"),
            "year + month"
        );

        // Mixed case
        assert_eq!(parse_current_in_expr("current YEAR - 1"), "YEAR - 1");

        // No CURRENT
        assert_eq!(parse_current_in_expr("year - 1"), "year - 1");
    }

    #[test]
    fn test_parse_at_modifier_with_current() {
        // CURRENT should be stripped from SET expression
        let result = parse_at_modifier("SET year = CURRENT year - 1").unwrap();
        assert_eq!(
            result,
            ContextModifier::Set("year".to_string(), "year - 1".to_string())
        );
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
        assert_eq!(extract_where_clause(sql3), Some("a = 1 AND b = 2".to_string()));
    }

    #[test]
    fn test_expand_modifiers_to_sql_chained_all() {
        // Chaining AT (ALL region) AT (ALL category) should correlate on remaining dims (year)
        let modifiers = vec![
            ContextModifier::All("region".to_string()),
            ContextModifier::All("category".to_string()),
        ];
        let group_by = vec!["year".to_string(), "region".to_string(), "category".to_string()];
        let expanded = expand_modifiers_to_sql("revenue", "SUM", &modifiers, "sales_v", Some("_outer"), None, &group_by);
        // Should correlate on year only (not grand total)
        assert_eq!(
            expanded,
            "(SELECT SUM(revenue) FROM sales_v _inner WHERE _inner.year = _outer.year)"
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
        let expanded = expand_modifiers_to_sql("revenue", "SUM", &modifiers, "sales_v", None, None, &group_by);
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
        let expanded = expand_modifiers_to_sql("revenue", "SUM", &modifiers, "sales_v", Some("_outer"), None, &group_by);
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
                },
                ViewMeasure {
                    column_name: "cost".to_string(),
                    expression: "SUM(expense)".to_string(),
                },
            ],
            base_query: "".to_string(),
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
            }],
            "SELECT year, SUM(amount) AS revenue FROM sales GROUP BY year",
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
        eprintln!("Result: {:?}", result);
        assert_eq!(result, Some(("sales_yearly".to_string(), Some("s".to_string()))));
    }

    // =========================================================================
    // JOIN Support Tests
    // =========================================================================

    #[test]
    fn test_extract_from_clause_info_simple() {
        let dialect = GenericDialect {};
        let sql = "SELECT * FROM orders";
        let statements = Parser::parse_sql(&dialect, sql).unwrap();
        if let Statement::Query(q) = &statements[0] {
            if let SetExpr::Select(s) = &*q.body {
                let info = extract_from_clause_info(s);
                assert_eq!(info.tables.len(), 1);
                assert!(info.tables.contains_key("orders"));
                assert_eq!(info.primary_table.as_ref().unwrap().name, "orders");
                assert!(!info.primary_table.as_ref().unwrap().has_alias);
            }
        }
    }

    #[test]
    fn test_extract_from_clause_info_with_alias() {
        let dialect = GenericDialect {};
        let sql = "SELECT * FROM orders o";
        let statements = Parser::parse_sql(&dialect, sql).unwrap();
        if let Statement::Query(q) = &statements[0] {
            if let SetExpr::Select(s) = &*q.body {
                let info = extract_from_clause_info(s);
                assert_eq!(info.tables.len(), 1);
                assert!(info.tables.contains_key("o"));
                let pt = info.primary_table.as_ref().unwrap();
                assert_eq!(pt.name, "orders");
                assert_eq!(pt.effective_name, "o");
                assert!(pt.has_alias);
            }
        }
    }

    #[test]
    fn test_extract_from_clause_info_join() {
        let dialect = GenericDialect {};
        let sql = "SELECT * FROM orders o JOIN customers c ON o.customer_id = c.id";
        let statements = Parser::parse_sql(&dialect, sql).unwrap();
        if let Statement::Query(q) = &statements[0] {
            if let SetExpr::Select(s) = &*q.body {
                let info = extract_from_clause_info(s);
                eprintln!("Tables: {:?}", info.tables);
                assert_eq!(info.tables.len(), 2);
                assert!(info.tables.contains_key("o"));
                assert!(info.tables.contains_key("c"));
                assert_eq!(info.tables.get("o").unwrap().name, "orders");
                assert_eq!(info.tables.get("c").unwrap().name, "customers");
            }
        }
    }

    #[test]
    fn test_extract_from_clause_info_multiple_joins() {
        let dialect = GenericDialect {};
        let sql = "SELECT * FROM orders o \
                   JOIN customers c ON o.customer_id = c.id \
                   LEFT JOIN products p ON o.product_id = p.id";
        let statements = Parser::parse_sql(&dialect, sql).unwrap();
        if let Statement::Query(q) = &statements[0] {
            if let SetExpr::Select(s) = &*q.body {
                let info = extract_from_clause_info(s);
                assert_eq!(info.tables.len(), 3);
                assert!(info.tables.contains_key("o"));
                assert!(info.tables.contains_key("c"));
                assert!(info.tables.contains_key("p"));
            }
        }
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
            }],
            "SELECT year, SUM(amount) AS revenue FROM orders GROUP BY year",
        );

        // Should find revenue in orders_v
        let (agg_fn, source, derived) = resolve_measure_source("revenue", "fallback");
        assert_eq!(agg_fn, "SUM");
        assert_eq!(source, "orders_v");
        assert!(derived.is_none()); // Not a derived measure

        // Unknown measure should fallback
        let (agg_fn2, source2, derived2) = resolve_measure_source("unknown", "fallback");
        assert_eq!(agg_fn2, "SUM");
        assert_eq!(source2, "fallback");
        assert!(derived2.is_none());
    }

    #[test]
    fn test_find_alias_for_view() {
        let dialect = GenericDialect {};
        let sql = "SELECT * FROM orders_v o JOIN customers c ON o.id = c.order_id";
        let statements = Parser::parse_sql(&dialect, sql).unwrap();
        if let Statement::Query(q) = &statements[0] {
            if let SetExpr::Select(s) = &*q.body {
                let info = extract_from_clause_info(s);

                // orders_v has alias "o"
                assert_eq!(find_alias_for_view(&info, "orders_v"), Some("o"));

                // customers has alias "c"
                assert_eq!(find_alias_for_view(&info, "customers"), Some("c"));

                // nonexistent table
                assert_eq!(find_alias_for_view(&info, "nonexistent"), None);
            }
        }
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
            }],
            "SELECT year, SUM(amount) AS revenue FROM orders GROUP BY year",
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
            }],
            "SELECT year, SUM(amount) AS revenue FROM orders GROUP BY year",
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
        assert_eq!(
            result,
            ContextModifier::All("YEAR(created_at)".to_string())
        );

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
    }
}
