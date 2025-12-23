//! Error types for yardstick

use thiserror::Error;

#[derive(Error, Debug)]
pub enum YardstickError {
    #[error("SQL parse error: {0}")]
    SqlParse(String),

    #[error("SQL generation error: {0}")]
    SqlGeneration(String),

    #[error("Measure not found: {0}")]
    MeasureNotFound(String),

    #[error("Invalid AT modifier: {0}")]
    InvalidAtModifier(String),

    #[error("Validation error: {0}")]
    Validation(String),

    #[error("AT modifier '{modifier}' not supported for non-decomposable measure '{measure}' (COUNT DISTINCT cannot be re-aggregated)")]
    NonDecomposableAtModifier { modifier: String, measure: String },
}

pub type Result<T> = std::result::Result<T, YardstickError>;
