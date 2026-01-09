//! Rust FFI bindings for DuckDB parser integration via C++
//!
//! This module provides Rust bindings that call the C++ parser functions
//! defined in yardstick_parser_ffi.cpp, which use DuckDB's native parser.
//!
//! Flow: Rust code -> calls extern "C" functions -> C++ uses duckdb::Parser -> returns C structs
//!
//! Memory model:
//! - C++ allocates memory using strdup/new
//! - Rust calls corresponding free functions to deallocate
//! - Safe wrappers handle cleanup via Drop trait or explicit cleanup

#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::ffi::{c_char, CStr, CString};
use std::ptr;

// =============================================================================
// C-compatible types matching yardstick_ffi.h
// =============================================================================

/// AT modifier type enum matching C definition
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum YardstickAtType {
    None = 0,
    AllGlobal = 1,
    AllDim = 2,
    Set = 3,
    Where = 4,
    Visible = 5,
}

/// Single AT modifier matching C definition
#[repr(C)]
#[derive(Debug)]
pub struct YardstickAtModifier {
    pub at_type: YardstickAtType,
    pub dimension: *const c_char,
    pub value: *const c_char,
}

impl Default for YardstickAtModifier {
    fn default() -> Self {
        Self {
            at_type: YardstickAtType::None,
            dimension: ptr::null(),
            value: ptr::null(),
        }
    }
}

/// Information about a single AGGREGATE() call
#[repr(C)]
#[derive(Debug)]
pub struct YardstickAggregateCall {
    pub measure_name: *const c_char,
    pub start_pos: u32,
    pub end_pos: u32,
    pub modifiers: *mut YardstickAtModifier,
    pub modifier_count: usize,
}

/// List of AGGREGATE() calls found in SQL
#[repr(C)]
#[derive(Debug)]
pub struct YardstickAggregateCallList {
    pub calls: *mut YardstickAggregateCall,
    pub count: usize,
    pub error: *const c_char,
}

/// Information about a single SELECT item
#[repr(C)]
#[derive(Debug)]
pub struct YardstickSelectItem {
    pub expression_sql: *const c_char,
    pub alias: *const c_char,
    pub start_pos: u32,
    pub end_pos: u32,
    pub is_aggregate: bool,
    pub is_star: bool,
    pub is_measure_ref: bool,
}

/// Information about a table in FROM clause
#[repr(C)]
#[derive(Debug)]
pub struct YardstickTableRef {
    pub table_name: *const c_char,
    pub alias: *const c_char,
    pub is_subquery: bool,
}

/// Full SELECT clause information
#[repr(C)]
#[derive(Debug)]
pub struct YardstickSelectInfo {
    pub items: *mut YardstickSelectItem,
    pub item_count: usize,
    pub tables: *mut YardstickTableRef,
    pub table_count: usize,
    pub primary_table: *const c_char,
    pub group_by_cols: *mut *const c_char,
    pub group_by_count: usize,
    pub has_group_by: bool,
    pub group_by_all: bool,
    pub where_clause: *const c_char,
    pub error: *const c_char,
}

/// Parsed expression information
#[repr(C)]
#[derive(Debug)]
pub struct YardstickExpressionInfo {
    pub sql: *const c_char,
    pub aggregate_func: *const c_char,
    pub inner_expr: *const c_char,
    pub is_aggregate: bool,
    pub is_identifier: bool,
    pub error: *const c_char,
}

/// Measure definition from CREATE VIEW AS MEASURE
#[repr(C)]
#[derive(Debug)]
pub struct YardstickMeasureDef {
    pub column_name: *const c_char,
    pub expression: *const c_char,
    pub aggregate_func: *const c_char,
    pub is_derived: bool,
}

/// Result from parsing CREATE VIEW with AS MEASURE
#[repr(C)]
#[derive(Debug)]
pub struct YardstickCreateViewInfo {
    pub is_measure_view: bool,
    pub view_name: *const c_char,
    pub clean_sql: *const c_char,
    pub measures: *mut YardstickMeasureDef,
    pub measure_count: usize,
    pub error: *const c_char,
}

/// Single replacement in SQL text
#[repr(C)]
#[derive(Debug)]
pub struct YardstickReplacement {
    pub start_pos: u32,
    pub end_pos: u32,
    pub replacement: *const c_char,
}

// =============================================================================
// Function pointer types for C++ functions (set at runtime to avoid link errors)
// =============================================================================

use std::sync::atomic::{AtomicPtr, Ordering};

type FnFindAggregates = unsafe extern "C" fn(*const c_char) -> *mut YardstickAggregateCallList;
type FnFreeAggregateList = unsafe extern "C" fn(*mut YardstickAggregateCallList);
type FnParseSelect = unsafe extern "C" fn(*const c_char) -> *mut YardstickSelectInfo;
type FnFreeSelectInfo = unsafe extern "C" fn(*mut YardstickSelectInfo);
type FnParseExpression = unsafe extern "C" fn(*const c_char) -> *mut YardstickExpressionInfo;
type FnFreeExpressionInfo = unsafe extern "C" fn(*mut YardstickExpressionInfo);
type FnParseCreateView = unsafe extern "C" fn(*const c_char) -> *mut YardstickCreateViewInfo;
type FnFreeCreateViewInfo = unsafe extern "C" fn(*mut YardstickCreateViewInfo);
type FnReplaceRange = unsafe extern "C" fn(*const c_char, u32, u32, *const c_char) -> *mut c_char;
type FnApplyReplacements = unsafe extern "C" fn(*const c_char, *const YardstickReplacement, usize) -> *mut c_char;
type FnQualifyExpression = unsafe extern "C" fn(*const c_char, *const c_char) -> *mut c_char;
type FnFreeString = unsafe extern "C" fn(*mut c_char);
type FnExpandAggregateCall = unsafe extern "C" fn(
    *const c_char, *const c_char, *const YardstickAtModifier, usize,
    *const c_char, *const c_char, *const c_char, *const *const c_char, usize
) -> *mut c_char;

// Static function pointers - set by C++ at init time
static FN_FIND_AGGREGATES: AtomicPtr<()> = AtomicPtr::new(ptr::null_mut());
static FN_FREE_AGGREGATE_LIST: AtomicPtr<()> = AtomicPtr::new(ptr::null_mut());
static FN_PARSE_SELECT: AtomicPtr<()> = AtomicPtr::new(ptr::null_mut());
static FN_FREE_SELECT_INFO: AtomicPtr<()> = AtomicPtr::new(ptr::null_mut());
static FN_PARSE_EXPRESSION: AtomicPtr<()> = AtomicPtr::new(ptr::null_mut());
static FN_FREE_EXPRESSION_INFO: AtomicPtr<()> = AtomicPtr::new(ptr::null_mut());
static FN_PARSE_CREATE_VIEW: AtomicPtr<()> = AtomicPtr::new(ptr::null_mut());
static FN_FREE_CREATE_VIEW_INFO: AtomicPtr<()> = AtomicPtr::new(ptr::null_mut());
static FN_REPLACE_RANGE: AtomicPtr<()> = AtomicPtr::new(ptr::null_mut());
static FN_APPLY_REPLACEMENTS: AtomicPtr<()> = AtomicPtr::new(ptr::null_mut());
static FN_QUALIFY_EXPRESSION: AtomicPtr<()> = AtomicPtr::new(ptr::null_mut());
static FN_FREE_STRING: AtomicPtr<()> = AtomicPtr::new(ptr::null_mut());
static FN_EXPAND_AGGREGATE_CALL: AtomicPtr<()> = AtomicPtr::new(ptr::null_mut());

/// Initialize function pointers - called by C++ at extension load time
#[no_mangle]
pub extern "C" fn yardstick_init_parser_ffi(
    find_aggregates: FnFindAggregates,
    free_aggregate_list: FnFreeAggregateList,
    parse_select: FnParseSelect,
    free_select_info: FnFreeSelectInfo,
    parse_expression: FnParseExpression,
    free_expression_info: FnFreeExpressionInfo,
    parse_create_view: FnParseCreateView,
    free_create_view_info: FnFreeCreateViewInfo,
    replace_range: FnReplaceRange,
    apply_replacements: FnApplyReplacements,
    qualify_expression: FnQualifyExpression,
    free_string: FnFreeString,
    expand_aggregate_call: FnExpandAggregateCall,
) {
    FN_FIND_AGGREGATES.store(find_aggregates as *mut (), Ordering::SeqCst);
    FN_FREE_AGGREGATE_LIST.store(free_aggregate_list as *mut (), Ordering::SeqCst);
    FN_PARSE_SELECT.store(parse_select as *mut (), Ordering::SeqCst);
    FN_FREE_SELECT_INFO.store(free_select_info as *mut (), Ordering::SeqCst);
    FN_PARSE_EXPRESSION.store(parse_expression as *mut (), Ordering::SeqCst);
    FN_FREE_EXPRESSION_INFO.store(free_expression_info as *mut (), Ordering::SeqCst);
    FN_PARSE_CREATE_VIEW.store(parse_create_view as *mut (), Ordering::SeqCst);
    FN_FREE_CREATE_VIEW_INFO.store(free_create_view_info as *mut (), Ordering::SeqCst);
    FN_REPLACE_RANGE.store(replace_range as *mut (), Ordering::SeqCst);
    FN_APPLY_REPLACEMENTS.store(apply_replacements as *mut (), Ordering::SeqCst);
    FN_QUALIFY_EXPRESSION.store(qualify_expression as *mut (), Ordering::SeqCst);
    FN_FREE_STRING.store(free_string as *mut (), Ordering::SeqCst);
    FN_EXPAND_AGGREGATE_CALL.store(expand_aggregate_call as *mut (), Ordering::SeqCst);
}

// Helper macros to call function pointers
macro_rules! call_ffi {
    ($ptr:expr, $type:ty, $($arg:expr),*) => {{
        let p = $ptr.load(Ordering::SeqCst);
        if p.is_null() {
            panic!("Parser FFI not initialized - call yardstick_init_parser_ffi first");
        }
        let f: $type = std::mem::transmute(p);
        f($($arg),*)
    }};
}

unsafe fn yardstick_find_aggregates(sql: *const c_char) -> *mut YardstickAggregateCallList {
    call_ffi!(FN_FIND_AGGREGATES, FnFindAggregates, sql)
}

unsafe fn yardstick_free_aggregate_list(list: *mut YardstickAggregateCallList) {
    call_ffi!(FN_FREE_AGGREGATE_LIST, FnFreeAggregateList, list)
}

unsafe fn yardstick_parse_select(sql: *const c_char) -> *mut YardstickSelectInfo {
    call_ffi!(FN_PARSE_SELECT, FnParseSelect, sql)
}

unsafe fn yardstick_free_select_info(info: *mut YardstickSelectInfo) {
    call_ffi!(FN_FREE_SELECT_INFO, FnFreeSelectInfo, info)
}

unsafe fn yardstick_parse_expression(expr: *const c_char) -> *mut YardstickExpressionInfo {
    call_ffi!(FN_PARSE_EXPRESSION, FnParseExpression, expr)
}

unsafe fn yardstick_free_expression_info(info: *mut YardstickExpressionInfo) {
    call_ffi!(FN_FREE_EXPRESSION_INFO, FnFreeExpressionInfo, info)
}

unsafe fn yardstick_parse_create_view(sql: *const c_char) -> *mut YardstickCreateViewInfo {
    call_ffi!(FN_PARSE_CREATE_VIEW, FnParseCreateView, sql)
}

unsafe fn yardstick_free_create_view_info(info: *mut YardstickCreateViewInfo) {
    call_ffi!(FN_FREE_CREATE_VIEW_INFO, FnFreeCreateViewInfo, info)
}

unsafe fn yardstick_replace_range(sql: *const c_char, start: u32, end: u32, replacement: *const c_char) -> *mut c_char {
    call_ffi!(FN_REPLACE_RANGE, FnReplaceRange, sql, start, end, replacement)
}

unsafe fn yardstick_apply_replacements(sql: *const c_char, replacements: *const YardstickReplacement, count: usize) -> *mut c_char {
    call_ffi!(FN_APPLY_REPLACEMENTS, FnApplyReplacements, sql, replacements, count)
}

unsafe fn yardstick_free_string(ptr: *mut c_char) {
    call_ffi!(FN_FREE_STRING, FnFreeString, ptr)
}

unsafe fn yardstick_expand_aggregate_call(
    measure_name: *const c_char,
    agg_func: *const c_char,
    modifiers: *const YardstickAtModifier,
    modifier_count: usize,
    table_name: *const c_char,
    outer_alias: *const c_char,
    outer_where: *const c_char,
    group_by_cols: *const *const c_char,
    group_by_count: usize,
) -> *mut c_char {
    call_ffi!(FN_EXPAND_AGGREGATE_CALL, FnExpandAggregateCall,
        measure_name, agg_func, modifiers, modifier_count,
        table_name, outer_alias, outer_where, group_by_cols, group_by_count)
}

// =============================================================================
// Safe Rust wrapper types
// =============================================================================

/// AT modifier type (safe Rust enum)
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AtType {
    None,
    AllGlobal,
    AllDim,
    Set,
    Where,
    Visible,
}

impl From<YardstickAtType> for AtType {
    fn from(c_type: YardstickAtType) -> Self {
        match c_type {
            YardstickAtType::None => AtType::None,
            YardstickAtType::AllGlobal => AtType::AllGlobal,
            YardstickAtType::AllDim => AtType::AllDim,
            YardstickAtType::Set => AtType::Set,
            YardstickAtType::Where => AtType::Where,
            YardstickAtType::Visible => AtType::Visible,
        }
    }
}

impl From<AtType> for YardstickAtType {
    fn from(rust_type: AtType) -> Self {
        match rust_type {
            AtType::None => YardstickAtType::None,
            AtType::AllGlobal => YardstickAtType::AllGlobal,
            AtType::AllDim => YardstickAtType::AllDim,
            AtType::Set => YardstickAtType::Set,
            AtType::Where => YardstickAtType::Where,
            AtType::Visible => YardstickAtType::Visible,
        }
    }
}

/// Safe wrapper for AT modifier
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AtModifier {
    pub modifier_type: AtType,
    pub dimension: Option<String>,
    pub value: Option<String>,
}

impl AtModifier {
    /// Create an ALL (global total) modifier
    pub fn all_global() -> Self {
        Self {
            modifier_type: AtType::AllGlobal,
            dimension: None,
            value: None,
        }
    }

    /// Create an ALL dimension modifier
    pub fn all_dim(dimension: impl Into<String>) -> Self {
        Self {
            modifier_type: AtType::AllDim,
            dimension: Some(dimension.into()),
            value: None,
        }
    }

    /// Create a SET modifier
    pub fn set(dimension: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            modifier_type: AtType::Set,
            dimension: Some(dimension.into()),
            value: Some(value.into()),
        }
    }

    /// Create a WHERE modifier
    pub fn where_clause(condition: impl Into<String>) -> Self {
        Self {
            modifier_type: AtType::Where,
            dimension: None,
            value: Some(condition.into()),
        }
    }

    /// Create a VISIBLE modifier
    pub fn visible() -> Self {
        Self {
            modifier_type: AtType::Visible,
            dimension: None,
            value: None,
        }
    }
}

/// Safe wrapper for aggregate call information
#[derive(Debug, Clone)]
pub struct AggregateCall {
    pub measure_name: String,
    pub start_pos: u32,
    pub end_pos: u32,
    pub modifiers: Vec<AtModifier>,
}

/// Safe wrapper for SELECT item information
#[derive(Debug, Clone)]
pub struct SelectItem {
    pub expression_sql: String,
    pub alias: Option<String>,
    pub start_pos: u32,
    pub end_pos: u32,
    pub is_aggregate: bool,
    pub is_star: bool,
    pub is_measure_ref: bool,
}

/// Safe wrapper for table reference information
#[derive(Debug, Clone)]
pub struct TableRef {
    pub table_name: String,
    pub alias: Option<String>,
    pub is_subquery: bool,
}

/// Safe wrapper for SELECT clause information
#[derive(Debug, Clone)]
pub struct SelectInfo {
    pub items: Vec<SelectItem>,
    pub tables: Vec<TableRef>,
    pub primary_table: Option<String>,
    pub group_by_cols: Vec<String>,
    pub has_group_by: bool,
    pub group_by_all: bool,
    pub where_clause: Option<String>,
}

/// Safe wrapper for expression information
#[derive(Debug, Clone)]
pub struct ExpressionInfo {
    pub sql: String,
    pub aggregate_func: Option<String>,
    pub inner_expr: Option<String>,
    pub is_aggregate: bool,
    pub is_identifier: bool,
}

/// Safe wrapper for measure definition
#[derive(Debug, Clone)]
pub struct MeasureDef {
    pub column_name: String,
    pub expression: String,
    pub aggregate_func: Option<String>,
    pub is_derived: bool,
}

/// Safe wrapper for CREATE VIEW info
#[derive(Debug, Clone)]
pub struct CreateViewInfo {
    pub is_measure_view: bool,
    pub view_name: Option<String>,
    pub clean_sql: Option<String>,
    pub measures: Vec<MeasureDef>,
}

/// Replacement operation (safe Rust type)
#[derive(Debug, Clone)]
pub struct Replacement {
    pub start_pos: u32,
    pub end_pos: u32,
    pub replacement: String,
}

// =============================================================================
// Helper functions
// =============================================================================

/// Convert C string to Rust Option<String>
///
/// # Safety
/// Caller must ensure ptr is valid or null
unsafe fn c_str_to_string(ptr: *const c_char) -> Option<String> {
    if ptr.is_null() {
        None
    } else {
        CStr::from_ptr(ptr).to_str().ok().map(|s| s.to_string())
    }
}


// =============================================================================
// Safe wrapper functions that call C++ via FFI
// =============================================================================

/// Find all AGGREGATE() calls in SQL using DuckDB's parser.
///
/// This calls the C++ implementation which uses DuckDB's native parser
/// to find AGGREGATE() function calls and their AT modifiers.
///
/// # Example
/// ```ignore
/// let calls = find_aggregates("SELECT AGGREGATE(revenue) AT (ALL) FROM sales")?;
/// assert_eq!(calls.len(), 1);
/// assert_eq!(calls[0].measure_name, "revenue");
/// ```
pub fn find_aggregates(sql: &str) -> Result<Vec<AggregateCall>, String> {
    let c_sql = CString::new(sql).map_err(|e| format!("Invalid SQL string: {e}"))?;

    unsafe {
        let list_ptr = yardstick_find_aggregates(c_sql.as_ptr());
        if list_ptr.is_null() {
            return Err("Failed to parse SQL".to_string());
        }

        let list = &*list_ptr;

        // Check for error
        if !list.error.is_null() {
            let error_msg = c_str_to_string(list.error).unwrap_or_else(|| "Unknown error".to_string());
            yardstick_free_aggregate_list(list_ptr);
            return Err(error_msg);
        }

        // Convert calls to Rust types
        let mut result = Vec::with_capacity(list.count);

        for i in 0..list.count {
            let call = &*list.calls.add(i);

            let measure_name = c_str_to_string(call.measure_name)
                .unwrap_or_default();

            // Convert modifiers
            let mut modifiers = Vec::with_capacity(call.modifier_count);
            for j in 0..call.modifier_count {
                let mod_ptr = call.modifiers.add(j);
                let modifier = &*mod_ptr;

                modifiers.push(AtModifier {
                    modifier_type: modifier.at_type.into(),
                    dimension: c_str_to_string(modifier.dimension),
                    value: c_str_to_string(modifier.value),
                });
            }

            result.push(AggregateCall {
                measure_name,
                start_pos: call.start_pos,
                end_pos: call.end_pos,
                modifiers,
            });
        }

        yardstick_free_aggregate_list(list_ptr);
        Ok(result)
    }
}

/// Parse SELECT statement and extract structure using DuckDB's parser.
///
/// Returns detailed information about SELECT items, tables, GROUP BY, and WHERE clauses.
///
/// # Example
/// ```ignore
/// let info = parse_select("SELECT region, SUM(amount) FROM sales GROUP BY region")?;
/// assert_eq!(info.items.len(), 2);
/// assert!(info.has_group_by);
/// ```
pub fn parse_select(sql: &str) -> Result<SelectInfo, String> {
    let c_sql = CString::new(sql).map_err(|e| format!("Invalid SQL string: {e}"))?;

    unsafe {
        let info_ptr = yardstick_parse_select(c_sql.as_ptr());
        if info_ptr.is_null() {
            return Err("Failed to parse SELECT".to_string());
        }

        let info = &*info_ptr;

        // Check for error
        if !info.error.is_null() {
            let error_msg = c_str_to_string(info.error).unwrap_or_else(|| "Unknown error".to_string());
            yardstick_free_select_info(info_ptr);
            return Err(error_msg);
        }

        // Convert items
        let mut items = Vec::with_capacity(info.item_count);
        for i in 0..info.item_count {
            let item = &*info.items.add(i);
            items.push(SelectItem {
                expression_sql: c_str_to_string(item.expression_sql).unwrap_or_default(),
                alias: c_str_to_string(item.alias),
                start_pos: item.start_pos,
                end_pos: item.end_pos,
                is_aggregate: item.is_aggregate,
                is_star: item.is_star,
                is_measure_ref: item.is_measure_ref,
            });
        }

        // Convert tables
        let mut tables = Vec::with_capacity(info.table_count);
        for i in 0..info.table_count {
            let table = &*info.tables.add(i);
            tables.push(TableRef {
                table_name: c_str_to_string(table.table_name).unwrap_or_default(),
                alias: c_str_to_string(table.alias),
                is_subquery: table.is_subquery,
            });
        }

        // Convert GROUP BY columns
        let mut group_by_cols = Vec::with_capacity(info.group_by_count);
        for i in 0..info.group_by_count {
            let col_ptr = *info.group_by_cols.add(i);
            if let Some(col) = c_str_to_string(col_ptr) {
                group_by_cols.push(col);
            }
        }

        let result = SelectInfo {
            items,
            tables,
            primary_table: c_str_to_string(info.primary_table),
            group_by_cols,
            has_group_by: info.has_group_by,
            group_by_all: info.group_by_all,
            where_clause: c_str_to_string(info.where_clause),
        };

        yardstick_free_select_info(info_ptr);
        Ok(result)
    }
}

/// Parse a single SQL expression using DuckDB's parser.
///
/// # Example
/// ```ignore
/// let info = parse_expression("SUM(amount)")?;
/// assert!(info.is_aggregate);
/// assert_eq!(info.aggregate_func, Some("SUM".to_string()));
/// ```
pub fn parse_expression(expr: &str) -> Result<ExpressionInfo, String> {
    let c_expr = CString::new(expr).map_err(|e| format!("Invalid expression string: {e}"))?;

    unsafe {
        let info_ptr = yardstick_parse_expression(c_expr.as_ptr());
        if info_ptr.is_null() {
            return Err("Failed to parse expression".to_string());
        }

        let info = &*info_ptr;

        // Check for error
        if !info.error.is_null() {
            let error_msg = c_str_to_string(info.error).unwrap_or_else(|| "Unknown error".to_string());
            yardstick_free_expression_info(info_ptr);
            return Err(error_msg);
        }

        let result = ExpressionInfo {
            sql: c_str_to_string(info.sql).unwrap_or_default(),
            aggregate_func: c_str_to_string(info.aggregate_func),
            inner_expr: c_str_to_string(info.inner_expr),
            is_aggregate: info.is_aggregate,
            is_identifier: info.is_identifier,
        };

        yardstick_free_expression_info(info_ptr);
        Ok(result)
    }
}

/// Parse CREATE VIEW with AS MEASURE syntax using DuckDB's parser.
///
/// # Example
/// ```ignore
/// let info = parse_create_view("CREATE VIEW metrics AS SELECT SUM(amount) AS revenue AS MEASURE FROM sales")?;
/// ```
pub fn parse_create_view(sql: &str) -> Result<CreateViewInfo, String> {
    let c_sql = CString::new(sql).map_err(|e| format!("Invalid SQL string: {e}"))?;

    unsafe {
        let info_ptr = yardstick_parse_create_view(c_sql.as_ptr());
        if info_ptr.is_null() {
            return Err("Failed to parse CREATE VIEW".to_string());
        }

        let info = &*info_ptr;

        // Check for error
        if !info.error.is_null() {
            let error_msg = c_str_to_string(info.error).unwrap_or_else(|| "Unknown error".to_string());
            yardstick_free_create_view_info(info_ptr);
            return Err(error_msg);
        }

        // Convert measures
        let mut measures = Vec::with_capacity(info.measure_count);
        for i in 0..info.measure_count {
            let measure = &*info.measures.add(i);
            measures.push(MeasureDef {
                column_name: c_str_to_string(measure.column_name).unwrap_or_default(),
                expression: c_str_to_string(measure.expression).unwrap_or_default(),
                aggregate_func: c_str_to_string(measure.aggregate_func),
                is_derived: measure.is_derived,
            });
        }

        let result = CreateViewInfo {
            is_measure_view: info.is_measure_view,
            view_name: c_str_to_string(info.view_name),
            clean_sql: c_str_to_string(info.clean_sql),
            measures,
        };

        yardstick_free_create_view_info(info_ptr);
        Ok(result)
    }
}

/// Replace a single range in SQL string.
///
/// # Example
/// ```ignore
/// let result = replace_range("SELECT foo FROM bar", 7, 10, "baz")?;
/// assert_eq!(result, "SELECT baz FROM bar");
/// ```
pub fn replace_range(sql: &str, start: u32, end: u32, replacement: &str) -> Result<String, String> {
    let c_sql = CString::new(sql).map_err(|e| format!("Invalid SQL string: {e}"))?;
    let c_replacement = CString::new(replacement).map_err(|e| format!("Invalid replacement string: {e}"))?;

    unsafe {
        let result_ptr = yardstick_replace_range(c_sql.as_ptr(), start, end, c_replacement.as_ptr());
        if result_ptr.is_null() {
            return Err("Failed to replace range".to_string());
        }

        let result = c_str_to_string(result_ptr).unwrap_or_default();
        yardstick_free_string(result_ptr);
        Ok(result)
    }
}

/// Apply multiple replacements to SQL (handles position adjustments).
///
/// Replacements are sorted and applied from end to start so positions remain valid.
///
/// # Example
/// ```ignore
/// let replacements = vec![
///     Replacement { start_pos: 7, end_pos: 10, replacement: "baz".to_string() },
///     Replacement { start_pos: 16, end_pos: 19, replacement: "qux".to_string() },
/// ];
/// let result = apply_replacements("SELECT foo FROM bar", &replacements)?;
/// assert_eq!(result, "SELECT baz FROM qux");
/// ```
pub fn apply_replacements(sql: &str, replacements: &[Replacement]) -> Result<String, String> {
    let c_sql = CString::new(sql).map_err(|e| format!("Invalid SQL string: {e}"))?;

    if replacements.is_empty() {
        return Ok(sql.to_string());
    }

    // Convert replacements to C structs
    // We need to keep CStrings alive for the duration of the call
    let c_replacement_strings: Vec<CString> = replacements
        .iter()
        .map(|r| CString::new(r.replacement.as_str()).unwrap_or_default())
        .collect();

    let c_replacements: Vec<YardstickReplacement> = replacements
        .iter()
        .zip(c_replacement_strings.iter())
        .map(|(r, cs)| YardstickReplacement {
            start_pos: r.start_pos,
            end_pos: r.end_pos,
            replacement: cs.as_ptr(),
        })
        .collect();

    unsafe {
        let result_ptr = yardstick_apply_replacements(
            c_sql.as_ptr(),
            c_replacements.as_ptr(),
            c_replacements.len(),
        );

        if result_ptr.is_null() {
            return Err("Failed to apply replacements".to_string());
        }

        let result = c_str_to_string(result_ptr).unwrap_or_default();
        yardstick_free_string(result_ptr);
        Ok(result)
    }
}

pub fn qualify_expression(expr: &str, qualifier: &str) -> Result<String, String> {
    let expr_ptr = CString::new(expr).map_err(|e| format!("Invalid expression string: {e}"))?;
    let qualifier_ptr = CString::new(qualifier).map_err(|e| format!("Invalid qualifier: {e}"))?;

    let fn_ptr = FN_QUALIFY_EXPRESSION.load(Ordering::SeqCst);
    if fn_ptr.is_null() {
        return Err("Parser FFI not initialized".to_string());
    }

    unsafe {
        let f: FnQualifyExpression = std::mem::transmute(fn_ptr);
        let result_ptr = f(expr_ptr.as_ptr(), qualifier_ptr.as_ptr());
        if result_ptr.is_null() {
            return Err("Failed to qualify expression".to_string());
        }
        let result = c_str_to_string(result_ptr).unwrap_or_default();
        yardstick_free_string(result_ptr);
        Ok(result)
    }
}

/// Expand a single AGGREGATE() call to SQL.
///
/// Generates a correlated subquery for the measure based on the aggregation function
/// and AT modifiers.
///
/// # Arguments
/// * `measure_name` - Name of the measure column
/// * `agg_func` - Aggregation function (SUM, COUNT, etc.)
/// * `modifiers` - AT modifiers to apply
/// * `table_name` - Source table name
/// * `outer_alias` - Optional alias for outer query correlation
/// * `outer_where` - Optional WHERE clause from outer query (for VISIBLE)
/// * `group_by_cols` - GROUP BY columns for correlation
pub fn expand_aggregate_call(
    measure_name: &str,
    agg_func: &str,
    modifiers: &[AtModifier],
    table_name: &str,
    outer_alias: Option<&str>,
    outer_where: Option<&str>,
    group_by_cols: &[String],
) -> Result<String, String> {
    let c_measure = CString::new(measure_name).map_err(|e| format!("Invalid measure name: {e}"))?;
    let c_agg = CString::new(agg_func).map_err(|e| format!("Invalid agg function: {e}"))?;
    let c_table = CString::new(table_name).map_err(|e| format!("Invalid table name: {e}"))?;

    let c_outer_alias = outer_alias.and_then(|s| CString::new(s).ok());
    let c_outer_where = outer_where.and_then(|s| CString::new(s).ok());

    // Convert group by columns
    let c_group_by: Vec<CString> = group_by_cols
        .iter()
        .filter_map(|s| CString::new(s.as_str()).ok())
        .collect();
    let c_group_by_ptrs: Vec<*const c_char> = c_group_by.iter().map(|cs| cs.as_ptr()).collect();

    // Convert modifiers - need to keep dimension/value CStrings alive
    let mut mod_dims: Vec<Option<CString>> = Vec::with_capacity(modifiers.len());
    let mut mod_vals: Vec<Option<CString>> = Vec::with_capacity(modifiers.len());

    for m in modifiers {
        mod_dims.push(m.dimension.as_ref().and_then(|s| CString::new(s.as_str()).ok()));
        mod_vals.push(m.value.as_ref().and_then(|s| CString::new(s.as_str()).ok()));
    }

    let c_modifiers: Vec<YardstickAtModifier> = modifiers
        .iter()
        .enumerate()
        .map(|(i, m)| YardstickAtModifier {
            at_type: m.modifier_type.clone().into(),
            dimension: mod_dims[i].as_ref().map(|cs| cs.as_ptr()).unwrap_or(ptr::null()),
            value: mod_vals[i].as_ref().map(|cs| cs.as_ptr()).unwrap_or(ptr::null()),
        })
        .collect();

    unsafe {
        let result_ptr = yardstick_expand_aggregate_call(
            c_measure.as_ptr(),
            c_agg.as_ptr(),
            if c_modifiers.is_empty() { ptr::null() } else { c_modifiers.as_ptr() },
            c_modifiers.len(),
            c_table.as_ptr(),
            c_outer_alias.as_ref().map(|cs| cs.as_ptr()).unwrap_or(ptr::null()),
            c_outer_where.as_ref().map(|cs| cs.as_ptr()).unwrap_or(ptr::null()),
            if c_group_by_ptrs.is_empty() { ptr::null() } else { c_group_by_ptrs.as_ptr() },
            c_group_by_ptrs.len(),
        );

        if result_ptr.is_null() {
            return Err("Failed to expand aggregate call".to_string());
        }

        let result = c_str_to_string(result_ptr).unwrap_or_default();
        yardstick_free_string(result_ptr);
        Ok(result)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_at_modifier_constructors() {
        let all_global = AtModifier::all_global();
        assert_eq!(all_global.modifier_type, AtType::AllGlobal);
        assert!(all_global.dimension.is_none());
        assert!(all_global.value.is_none());

        let all_dim = AtModifier::all_dim("region");
        assert_eq!(all_dim.modifier_type, AtType::AllDim);
        assert_eq!(all_dim.dimension, Some("region".to_string()));

        let set_mod = AtModifier::set("year", "2024");
        assert_eq!(set_mod.modifier_type, AtType::Set);
        assert_eq!(set_mod.dimension, Some("year".to_string()));
        assert_eq!(set_mod.value, Some("2024".to_string()));

        let where_mod = AtModifier::where_clause("status = 'active'");
        assert_eq!(where_mod.modifier_type, AtType::Where);
        assert_eq!(where_mod.value, Some("status = 'active'".to_string()));

        let visible = AtModifier::visible();
        assert_eq!(visible.modifier_type, AtType::Visible);
    }

    #[test]
    fn test_at_type_conversion() {
        assert_eq!(AtType::from(YardstickAtType::None), AtType::None);
        assert_eq!(AtType::from(YardstickAtType::AllGlobal), AtType::AllGlobal);
        assert_eq!(AtType::from(YardstickAtType::AllDim), AtType::AllDim);
        assert_eq!(AtType::from(YardstickAtType::Set), AtType::Set);
        assert_eq!(AtType::from(YardstickAtType::Where), AtType::Where);
        assert_eq!(AtType::from(YardstickAtType::Visible), AtType::Visible);

        assert_eq!(YardstickAtType::from(AtType::None), YardstickAtType::None);
        assert_eq!(YardstickAtType::from(AtType::AllGlobal), YardstickAtType::AllGlobal);
    }

    #[test]
    fn test_replacement_struct() {
        let replacement = Replacement {
            start_pos: 7,
            end_pos: 10,
            replacement: "baz".to_string(),
        };
        assert_eq!(replacement.start_pos, 7);
        assert_eq!(replacement.end_pos, 10);
        assert_eq!(replacement.replacement, "baz");
    }

    // Note: The following tests require the C++ library to be linked.
    // They are marked as ignore for unit testing but can be run with integration tests.

    #[test]
    #[ignore = "requires C++ library to be linked"]
    fn test_find_aggregates_simple() {
        let calls = find_aggregates("SELECT AGGREGATE(revenue) FROM sales").unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].measure_name, "revenue");
        assert!(calls[0].modifiers.is_empty());
    }

    #[test]
    #[ignore = "requires C++ library to be linked"]
    fn test_find_aggregates_with_at() {
        let calls = find_aggregates("SELECT AGGREGATE(revenue) AT (ALL) FROM sales").unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].modifiers.len(), 1);
        assert_eq!(calls[0].modifiers[0].modifier_type, AtType::AllGlobal);
    }

    #[test]
    #[ignore = "requires C++ library to be linked"]
    fn test_parse_select() {
        let info = parse_select("SELECT region, SUM(amount) FROM sales GROUP BY region").unwrap();
        assert_eq!(info.items.len(), 2);
        assert_eq!(info.tables.len(), 1);
        assert!(info.has_group_by);
        assert!(!info.group_by_all);
        assert_eq!(info.group_by_cols.len(), 1);
    }

    #[test]
    #[ignore = "requires C++ library to be linked"]
    fn test_parse_expression() {
        let info = parse_expression("SUM(amount)").unwrap();
        assert!(info.is_aggregate);
        assert_eq!(info.aggregate_func, Some("SUM".to_string()));
        assert_eq!(info.inner_expr, Some("amount".to_string()));
    }

    #[test]
    #[ignore = "requires C++ library to be linked"]
    fn test_replace_range() {
        let result = replace_range("SELECT foo FROM bar", 7, 10, "baz").unwrap();
        assert_eq!(result, "SELECT baz FROM bar");
    }

    #[test]
    #[ignore = "requires C++ library to be linked"]
    fn test_apply_replacements() {
        let replacements = vec![
            Replacement {
                start_pos: 7,
                end_pos: 10,
                replacement: "baz".to_string(),
            },
        ];
        let result = apply_replacements("SELECT foo FROM bar", &replacements).unwrap();
        assert_eq!(result, "SELECT baz FROM bar");
    }

    #[test]
    #[ignore = "requires C++ library to be linked"]
    fn test_qualify_expression() {
        let result = qualify_expression("year between date '2023-01-01' and date '2025-01-01'", "_inner")
            .unwrap();
        assert_eq!(
            result,
            "_inner.year BETWEEN DATE '2023-01-01' AND DATE '2025-01-01'"
        );
    }

    #[test]
    #[ignore = "requires C++ library to be linked"]
    fn test_expand_aggregate_call() {
        let result = expand_aggregate_call(
            "revenue",
            "SUM",
            &[],
            "sales",
            None,
            None,
            &[],
        ).unwrap();
        assert!(result.contains("SUM"));
        assert!(result.contains("revenue"));
        assert!(result.contains("sales"));
    }
}
