//! SQL generation and query rewriting

mod generator;
pub mod measures;
mod rewriter;

pub use generator::{SemanticQuery, SqlGenerator};
pub use measures::{
    // Core types
    AggregateExpandResult, ContextModifier, CreateViewResult, MeasureView, ViewMeasure,
    // Detection functions
    has_aggregate_function, has_as_measure, has_at_syntax, has_curly_brace_measure,
    // Processing functions
    expand_aggregate, expand_aggregate_with_at, expand_curly_braces, process_create_view,
    // Phase 2: CTE generation
    expand_aggregate_with_cte, CteConfig, CteExpandResult,
    // Phase 6: Fan-out prevention
    apply_symmetric_aggregate, detect_fan_out_in_query, expand_aggregate_with_symmetric,
    // Measure lookup
    get_measure_aggregation,
};
pub use rewriter::QueryRewriter;
