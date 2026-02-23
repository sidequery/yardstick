//! C FFI bindings for yardstick-rs
//!
//! Exposes the Measures in SQL functionality to C/C++ consumers like the DuckDB extension.
//!
//! Safety: These functions are `extern "C"` and expect valid C strings.
//! Callers must ensure pointers are valid. Documented in header.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr;

use crate::sql::{
    drop_measure_view_from_sql, expand_aggregate_with_at, expand_curly_braces,
    get_measure_aggregation, has_aggregate_function, has_as_measure, has_at_syntax,
    has_curly_brace_measure, process_create_view,
    has_implicit_measure_refs, has_measure_at_refs,
};

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

/// Check if SQL contains AGGREGATE() or implicit measure references
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
        || has_implicit_measure_refs(sql_str)
        || has_measure_at_refs(sql_str)
}

/// Drop a measure view from the catalog if the SQL is a DROP VIEW statement
#[no_mangle]
pub extern "C" fn yardstick_drop_measure_view_from_sql(sql: *const c_char) -> bool {
    if sql.is_null() {
        return false;
    }

    let sql_str = unsafe {
        match CStr::from_ptr(sql).to_str() {
            Ok(s) => s,
            Err(_) => return false,
        }
    };

    drop_measure_view_from_sql(sql_str)
}

/// Check if SQL contains curly brace measure syntax: `{column}`
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
        view_name: result
            .view_name
            .map(|s| to_c_string(&s))
            .unwrap_or(ptr::null_mut()),
        clean_sql: to_c_string(&result.clean_sql),
        error: result
            .error
            .map(|s| to_c_string(&s))
            .unwrap_or(ptr::null_mut()),
    }
}

/// Expand AGGREGATE() function calls in a SELECT statement
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

    let result = expand_aggregate_with_at(sql_str);

    YardstickAggregateResult {
        had_aggregate: result.had_aggregate,
        expanded_sql: to_c_string(&result.expanded_sql),
        error: result
            .error
            .map(|s| to_c_string(&s))
            .unwrap_or(ptr::null_mut()),
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

/// Free a string returned by yardstick functions
#[no_mangle]
pub extern "C" fn yardstick_free(ptr: *mut c_char) {
    if !ptr.is_null() {
        unsafe {
            drop(CString::from_raw(ptr));
        }
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

// Helper: convert Rust string to C string
fn to_c_string(s: &str) -> *mut c_char {
    match CString::new(s) {
        Ok(cs) => cs.into_raw(),
        Err(_) => ptr::null_mut(),
    }
}
