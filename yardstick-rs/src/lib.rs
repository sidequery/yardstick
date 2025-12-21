//! Yardstick: Measures in SQL for DuckDB
//!
//! Implementation of Julian Hyde's "Measures in SQL" paper (arXiv:2406.00251).

pub mod error;
pub mod ffi;
pub mod parser_ffi;
pub mod sql;

pub use error::{Result, YardstickError};
