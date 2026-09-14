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
use std::cell::RefCell;

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
    pub native_parsed: bool,
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
    pub contains_subquery: bool,
    pub reference_column: *const c_char,
    pub reference_qualifier: *const c_char,
    pub subquery_dimensions: *const *const c_char,
    pub subquery_dimension_count: usize,
}

/// Information about a table in FROM clause
#[repr(C)]
#[derive(Debug)]
pub struct YardstickTableRef {
    pub table_name: *const c_char,
    pub alias: *const c_char,
    pub is_subquery: bool,
    pub schema_qualified: bool,
}

#[repr(C)]
pub struct YardstickQueryScope {
    pub start_pos: u32,
    pub end_pos: u32,
    pub visible_ctes: *const *const c_char,
    pub visible_cte_count: usize,
}

#[repr(C)]
pub struct YardstickQueryScopeList {
    pub scopes: *const YardstickQueryScope,
    pub count: usize,
}

pub struct QueryScope {
    pub start: usize,
    pub end: usize,
    pub visible_ctes: Vec<String>,
}

thread_local! {
    static QUERY_CTES: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
}

/// Carry enclosing CTE visibility when a native query body is lowered alone.
/// Restoring on Drop also isolates recursive parser callbacks and unwinding.
pub struct QueryScopeGuard(Vec<String>);

impl QueryScopeGuard {
    pub fn enter(ctes: &[String]) -> Self {
        Self(QUERY_CTES.with(|current| {
            let previous = current.borrow().clone();
            current.borrow_mut().extend_from_slice(ctes);
            previous
        }))
    }
}

impl Drop for QueryScopeGuard {
    fn drop(&mut self) {
        QUERY_CTES.with(|current| current.replace(std::mem::take(&mut self.0)));
    }
}

/// Native shorthand operand and its exact source span.
#[repr(C)]
#[derive(Debug)]
pub struct YardstickMeasureReference {
    pub column: *const c_char,
    pub qualifier: *const c_char,
    pub start_pos: u32,
    pub end_pos: u32,
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
    pub native_parsed: bool,
    pub at_references: *mut YardstickMeasureReference,
    pub at_reference_count: usize,
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
    pub is_scalar: bool,
}

/// Measure definition from CREATE VIEW AS MEASURE
#[repr(C)]
#[derive(Debug)]
pub struct YardstickMeasureDef {
    pub column_name: *const c_char,
    pub expression: *const c_char,
    pub aggregate_func: *const c_char,
    pub is_derived: bool,
    pub expr_start: u32,
    pub name_end: u32,
    pub alias_sql: *const c_char,
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
    pub native_parsed: bool,
    pub metadata_query_sql: *const c_char,
    pub requires_binding: bool,
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

type FnFindQueryScopes = unsafe extern "C" fn(*const c_char) -> *mut YardstickQueryScopeList;
type FnFreeQueryScopes = unsafe extern "C" fn(*mut YardstickQueryScopeList);
static FN_FIND_QUERY_SCOPES: AtomicPtr<()> = AtomicPtr::new(ptr::null_mut());
static FN_FREE_QUERY_SCOPES: AtomicPtr<()> = AtomicPtr::new(ptr::null_mut());

/// None preserves the compatibility frontend when the native grammar is unavailable.
pub fn find_query_scopes(sql: &str) -> Option<Vec<QueryScope>> {
    let find = FN_FIND_QUERY_SCOPES.load(Ordering::SeqCst);
    let free = FN_FREE_QUERY_SCOPES.load(Ordering::SeqCst);
    if find.is_null() || free.is_null() {
        return None;
    }
    let sql = CString::new(sql).ok()?;
    unsafe {
        let find: FnFindQueryScopes = std::mem::transmute(find);
        let free: FnFreeQueryScopes = std::mem::transmute(free);
        let list = find(sql.as_ptr());
        if list.is_null() {
            return None;
        }
        let mut scopes = Vec::with_capacity((*list).count);
        for index in 0..(*list).count {
            let scope = &*(*list).scopes.add(index);
            let mut visible_ctes = Vec::with_capacity(scope.visible_cte_count);
            for cte in 0..scope.visible_cte_count {
                visible_ctes.push(CStr::from_ptr(*scope.visible_ctes.add(cte)).to_string_lossy().into_owned());
            }
            scopes.push(QueryScope {
                start: scope.start_pos as usize,
                end: scope.end_pos as usize,
                visible_ctes,
            });
        }
        free(list);
        Some(scopes)
    }
}

#[repr(C)]
pub struct YardstickCurrentReference {
    pub dimension: *const c_char,
    pub start_pos: u32,
    pub end_pos: u32,
}

#[repr(C)]
pub struct YardstickCurrentReferenceList {
    pub references: *mut YardstickCurrentReference,
    pub count: usize,
    pub error: *const c_char,
}

type FnFindCurrentReferences = unsafe extern "C" fn(*const c_char) -> *mut YardstickCurrentReferenceList;
type FnFreeCurrentReferences = unsafe extern "C" fn(*mut YardstickCurrentReferenceList);
static FN_FIND_CURRENT_REFERENCES: AtomicPtr<()> = AtomicPtr::new(ptr::null_mut());
static FN_FREE_CURRENT_REFERENCES: AtomicPtr<()> = AtomicPtr::new(ptr::null_mut());
type FnCurrentWhereIsSingleValued = unsafe extern "C" fn(*const c_char, *const c_char, *const c_char) -> i32;
static FN_CURRENT_WHERE_IS_SINGLE_VALUED: AtomicPtr<()> = AtomicPtr::new(ptr::null_mut());
type FnExpressionsEqual = unsafe extern "C" fn(*const c_char, *const c_char) -> i32;
static FN_EXPRESSIONS_EQUAL: AtomicPtr<()> = AtomicPtr::new(ptr::null_mut());
type FnRewriteVisibleFilter = unsafe extern "C" fn(
    *const c_char, *const c_char, *const *const c_char, *const *const c_char, usize, *mut *mut c_char,
) -> *mut c_char;
static FN_REWRITE_VISIBLE_FILTER: AtomicPtr<()> = AtomicPtr::new(ptr::null_mut());

/// Rebind the measure relation while retaining nested and ancestor references.
/// None keeps the compatibility frontend when native parsing is unavailable.
pub fn rewrite_visible_filter(
    expression: &str,
    local_alias: Option<&str>,
    dimension_expressions: &std::collections::HashMap<String, String>,
) -> Option<Result<String, String>> {
    let function = FN_REWRITE_VISIBLE_FILTER.load(Ordering::SeqCst);
    if function.is_null() || FN_FREE_STRING.load(Ordering::SeqCst).is_null() {
        return None;
    }
    let expression = CString::new(expression).ok()?;
    let local_alias = CString::new(local_alias.unwrap_or("")).ok()?;
    let entries = dimension_expressions.iter()
        .map(|(name, expression)| Some((CString::new(name.as_str()).ok()?, CString::new(expression.as_str()).ok()?)))
        .collect::<Option<Vec<_>>>()?;
    let names: Vec<_> = entries.iter().map(|(name, _)| name.as_ptr()).collect();
    let expressions: Vec<_> = entries.iter().map(|(_, expression)| expression.as_ptr()).collect();
    unsafe {
        let function: FnRewriteVisibleFilter = std::mem::transmute(function);
        let mut error = ptr::null_mut();
        let result = function(expression.as_ptr(), local_alias.as_ptr(), names.as_ptr(), expressions.as_ptr(), names.len(), &mut error);
        if !error.is_null() {
            let text = CStr::from_ptr(error).to_string_lossy().into_owned();
            yardstick_free_string(error);
            if !result.is_null() {
                yardstick_free_string(result);
            }
            return Some(Err(text));
        }
        if result.is_null() {
            return None;
        }
        let text = CStr::from_ptr(result).to_string_lossy().into_owned();
        yardstick_free_string(result);
        Some(Ok(text))
    }
}

pub fn expressions_equal(left: &str, right: &str) -> Option<bool> {
    let function = FN_EXPRESSIONS_EQUAL.load(Ordering::SeqCst);
    if function.is_null() {
        return None;
    }
    let (Ok(left), Ok(right)) = (CString::new(left), CString::new(right)) else {
        return Some(false);
    };
    let result = unsafe {
        let function: FnExpressionsEqual = std::mem::transmute(function);
        function(left.as_ptr(), right.as_ptr())
    };
    (result >= 0).then_some(result == 1)
}

pub fn current_where_is_single_valued(predicate: &str, dimension: &str, qualifier: Option<&str>) -> Option<bool> {
    let function = FN_CURRENT_WHERE_IS_SINGLE_VALUED.load(Ordering::SeqCst);
    if function.is_null() {
        return None;
    }
    let predicate = CString::new(predicate).ok()?;
    let dimension = CString::new(dimension).ok()?;
    let qualifier = CString::new(qualifier.unwrap_or("")).ok()?;
    let result = unsafe {
        let function: FnCurrentWhereIsSingleValued = std::mem::transmute(function);
        function(predicate.as_ptr(), dimension.as_ptr(), qualifier.as_ptr())
    };
    (result >= 0).then_some(result == 1)
}

pub struct CurrentReference {
    pub dimension: String,
    pub start_pos: usize,
    pub end_pos: usize,
}

/// None means the native grammar is unavailable. Native parse errors remain
/// errors instead of being reinterpreted by the compatibility scanner.
pub fn find_current_references(expression: &str) -> Option<Result<Vec<CurrentReference>, String>> {
    let find = FN_FIND_CURRENT_REFERENCES.load(Ordering::SeqCst);
    let free = FN_FREE_CURRENT_REFERENCES.load(Ordering::SeqCst);
    if find.is_null() || free.is_null() {
        return None;
    }
    let expression_c = match CString::new(expression) {
        Ok(expression) => expression,
        Err(error) => return Some(Err(error.to_string())),
    };
    unsafe {
        let find: FnFindCurrentReferences = std::mem::transmute(find);
        let free: FnFreeCurrentReferences = std::mem::transmute(free);
        let list = find(expression_c.as_ptr());
        if list.is_null() {
            return None;
        }
        let result = if !(*list).error.is_null() {
            Err(CStr::from_ptr((*list).error).to_string_lossy().into_owned())
        } else {
            let mut references = Vec::with_capacity((*list).count);
            for index in 0..(*list).count {
                let reference = &*(*list).references.add(index);
                references.push(CurrentReference {
                    dimension: CStr::from_ptr(reference.dimension).to_string_lossy().into_owned(),
                    start_pos: reference.start_pos as usize,
                    end_pos: reference.end_pos as usize,
                });
            }
            Ok(references)
        };
        free(list);
        Some(result)
    }
}

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
type FnInlineOrderBySubqueryAliases = unsafe extern "C" fn(*const c_char) -> *mut c_char;
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
static FN_INLINE_ORDER_BY_SUBQUERY_ALIASES: AtomicPtr<()> = AtomicPtr::new(ptr::null_mut());
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
    inline_order_by_subquery_aliases: FnInlineOrderBySubqueryAliases,
    free_string: FnFreeString,
    expand_aggregate_call: FnExpandAggregateCall,
    find_current_references: FnFindCurrentReferences,
    free_current_references: FnFreeCurrentReferences,
    current_where_is_single_valued: FnCurrentWhereIsSingleValued,
    expressions_equal: FnExpressionsEqual,
    find_query_scopes: FnFindQueryScopes,
    free_query_scopes: FnFreeQueryScopes,
    rewrite_visible_filter: FnRewriteVisibleFilter,
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
    FN_INLINE_ORDER_BY_SUBQUERY_ALIASES.store(inline_order_by_subquery_aliases as *mut (), Ordering::SeqCst);
    FN_FREE_STRING.store(free_string as *mut (), Ordering::SeqCst);
    FN_EXPAND_AGGREGATE_CALL.store(expand_aggregate_call as *mut (), Ordering::SeqCst);
    FN_FIND_CURRENT_REFERENCES.store(find_current_references as *mut (), Ordering::SeqCst);
    FN_FREE_CURRENT_REFERENCES.store(free_current_references as *mut (), Ordering::SeqCst);
    FN_CURRENT_WHERE_IS_SINGLE_VALUED.store(current_where_is_single_valued as *mut (), Ordering::SeqCst);
    FN_EXPRESSIONS_EQUAL.store(expressions_equal as *mut (), Ordering::SeqCst);
    FN_FIND_QUERY_SCOPES.store(find_query_scopes as *mut (), Ordering::SeqCst);
    FN_FREE_QUERY_SCOPES.store(free_query_scopes as *mut (), Ordering::SeqCst);
    FN_REWRITE_VISIBLE_FILTER.store(rewrite_visible_filter as *mut (), Ordering::SeqCst);
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

unsafe fn yardstick_inline_order_by_subquery_aliases(sql: *const c_char) -> *mut c_char {
    call_ffi!(FN_INLINE_ORDER_BY_SUBQUERY_ALIASES, FnInlineOrderBySubqueryAliases, sql)
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
    pub contains_subquery: bool,
    pub reference_column: Option<String>,
    pub reference_qualifier: Option<String>,
    pub subquery_dimensions: Vec<String>,
}

/// Safe wrapper for table reference information
#[derive(Debug, Clone)]
pub struct TableRef {
    pub table_name: String,
    pub alias: Option<String>,
    pub is_subquery: bool,
}

/// Safe wrapper for a native shorthand operand.
#[derive(Debug, Clone)]
pub struct MeasureReference {
    pub column: String,
    pub qualifier: Option<String>,
    pub start_pos: u32,
    pub end_pos: u32,
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
    pub native_parsed: bool,
    pub at_references: Vec<MeasureReference>,
}

/// Safe wrapper for expression information
#[derive(Debug, Clone)]
pub struct ExpressionInfo {
    pub sql: String,
    pub aggregate_func: Option<String>,
    pub inner_expr: Option<String>,
    pub is_aggregate: bool,
    pub is_identifier: bool,
    /// Syntactic independence from column references and subqueries, not foldability.
    pub is_scalar: bool,
}

/// Safe wrapper for measure definition
#[derive(Debug, Clone)]
pub struct MeasureDef {
    pub column_name: String,
    pub expression: String,
    pub aggregate_func: Option<String>,
    pub is_derived: bool,
    pub expr_start: u32,
    pub name_end: u32,
    pub alias_sql: String,
}

/// Safe wrapper for CREATE VIEW info
#[derive(Debug, Clone)]
pub struct CreateViewInfo {
    pub is_measure_view: bool,
    pub view_name: Option<String>,
    pub clean_sql: Option<String>,
    pub measures: Vec<MeasureDef>,
    pub native_parsed: bool,
    pub error: Option<String>,
    pub metadata_query_sql: Option<String>,
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
    find_aggregates_with_source(sql)
        .map(|(calls, _)| calls)
        .map_err(|error| error.message)
}

#[derive(Debug)]
pub(crate) struct AggregateParseError {
    pub message: String,
    pub native_parsed: bool,
}

impl AggregateParseError {
    fn compatibility(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            native_parsed: false,
        }
    }
}

pub(crate) fn find_aggregates_with_source(
    sql: &str,
) -> Result<(Vec<AggregateCall>, bool), AggregateParseError> {
    if FN_FIND_AGGREGATES.load(Ordering::SeqCst).is_null() {
        return Err(AggregateParseError::compatibility("Parser FFI not initialized"));
    }
    let c_sql = CString::new(sql)
        .map_err(|e| AggregateParseError::compatibility(format!("Invalid SQL string: {e}")))?;

    unsafe {
        let list_ptr = yardstick_find_aggregates(c_sql.as_ptr());
        if list_ptr.is_null() {
            return Err(AggregateParseError::compatibility("Failed to parse SQL"));
        }

        let list = &*list_ptr;

        // Check for error
        if !list.error.is_null() {
            let error = AggregateParseError {
                message: c_str_to_string(list.error).unwrap_or_else(|| "Unknown error".to_string()),
                native_parsed: list.native_parsed,
            };
            yardstick_free_aggregate_list(list_ptr);
            return Err(error);
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

        let native_parsed = list.native_parsed;
        yardstick_free_aggregate_list(list_ptr);
        Ok((result, native_parsed))
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
    if FN_PARSE_SELECT.load(Ordering::SeqCst).is_null() {
        return Err("Parser FFI not initialized".to_string());
    }
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
                contains_subquery: item.contains_subquery,
                reference_column: c_str_to_string(item.reference_column),
                reference_qualifier: c_str_to_string(item.reference_qualifier),
                subquery_dimensions: (0..item.subquery_dimension_count)
                    .filter_map(|i| c_str_to_string(*item.subquery_dimensions.add(i)))
                    .collect(),
            });
        }

        // Convert tables
        let mut tables = Vec::with_capacity(info.table_count);
        for i in 0..info.table_count {
            let table = &*info.tables.add(i);
            let table_name = c_str_to_string(table.table_name).unwrap_or_default();
            let inherited_cte = info.native_parsed && !table.schema_qualified &&
                QUERY_CTES.with(|ctes| ctes.borrow().iter().any(|cte| cte.eq_ignore_ascii_case(&table_name)));
            tables.push(TableRef {
                table_name,
                alias: c_str_to_string(table.alias),
                is_subquery: table.is_subquery || inherited_cte,
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

        let mut at_references = Vec::with_capacity(info.at_reference_count);
        for i in 0..info.at_reference_count {
            let reference = &*info.at_references.add(i);
            at_references.push(MeasureReference {
                column: c_str_to_string(reference.column).unwrap_or_default(),
                qualifier: c_str_to_string(reference.qualifier),
                start_pos: reference.start_pos,
                end_pos: reference.end_pos,
            });
        }

        let result = SelectInfo {
            items,
            tables,
            primary_table: c_str_to_string(info.primary_table),
            group_by_cols,
            has_group_by: info.has_group_by,
            group_by_all: info.group_by_all,
            where_clause: c_str_to_string(info.where_clause),
            native_parsed: info.native_parsed,
            at_references,
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
    if FN_PARSE_EXPRESSION.load(Ordering::SeqCst).is_null() {
        return Err("Parser FFI not initialized".to_string());
    }
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
            is_scalar: info.is_scalar,
        };

        yardstick_free_expression_info(info_ptr);
        Ok(result)
    }
}

/// Parse CREATE VIEW with AS MEASURE syntax using DuckDB's parser.
///
/// # Example
/// ```ignore
/// let info = parse_create_view("CREATE VIEW metrics AS SELECT SUM(amount) AS MEASURE revenue FROM sales")?;
/// ```
pub fn parse_create_view(sql: &str) -> Result<CreateViewInfo, String> {
    if FN_PARSE_CREATE_VIEW.load(Ordering::SeqCst).is_null() {
        return Err("Parser FFI not initialized".to_string());
    }
    let c_sql = CString::new(sql).map_err(|e| format!("Invalid SQL string: {e}"))?;

    unsafe {
        let info_ptr = yardstick_parse_create_view(c_sql.as_ptr());
        if info_ptr.is_null() {
            return Err("Failed to parse CREATE VIEW".to_string());
        }

        let info = &*info_ptr;

        // Check for error
        if !info.error.is_null() && !info.native_parsed {
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
                expr_start: measure.expr_start,
                name_end: measure.name_end,
                alias_sql: c_str_to_string(measure.alias_sql).unwrap_or_default(),
            });
        }

        let result = CreateViewInfo {
            is_measure_view: info.is_measure_view,
            view_name: c_str_to_string(info.view_name),
            clean_sql: c_str_to_string(info.clean_sql),
            measures,
            native_parsed: info.native_parsed,
            error: c_str_to_string(info.error),
            metadata_query_sql: c_str_to_string(info.metadata_query_sql),
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

pub fn inline_order_by_subquery_aliases(sql: &str) -> Option<String> {
    let fn_ptr = FN_INLINE_ORDER_BY_SUBQUERY_ALIASES.load(Ordering::SeqCst);
    if fn_ptr.is_null() {
        return None;
    }

    let c_sql = CString::new(sql).ok()?;
    unsafe {
        let result_ptr = yardstick_inline_order_by_subquery_aliases(c_sql.as_ptr());
        if result_ptr.is_null() {
            return None;
        }

        let result = c_str_to_string(result_ptr).unwrap_or_default();
        yardstick_free_string(result_ptr);
        Some(result)
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
    fn test_find_aggregates_with_comments_in_at_chain() {
        let sql = "SELECT AGGREGATE(revenue) /* keep */ AT (ALL) FROM sales";
        let calls = find_aggregates(sql).unwrap();
        assert_eq!(calls.len(), 1);
        let call = &calls[0];
        assert_eq!(
            &sql[call.start_pos as usize..call.end_pos as usize],
            "AGGREGATE(revenue) /* keep */ AT (ALL)"
        );
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
    fn test_parse_select_item_positions_ignore_comment_tokens() {
        let sql = "SELECT region /* , fake comma FROM fake */, AGGREGATE(revenue) FROM sales";
        let info = parse_select(sql).unwrap();
        assert_eq!(info.items.len(), 2);
        let first = &info.items[0];
        assert_eq!(
            sql[first.start_pos as usize..first.end_pos as usize].trim(),
            "region /* , fake comma FROM fake */"
        );
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
