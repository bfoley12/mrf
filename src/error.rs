use thiserror::Error;

#[derive(Error, Debug)]
pub enum MrfError{
    #[error("Asymmetric matrix: differing values found at ({row}, {col}) and ({col}, {row})")]
    AsymmetricMatrix { row: usize, col: usize },
    #[error("Dimension Mismatch: expected: {expected} got: {got}")]
    DimensionMismatch { expected: usize, got: usize },
    #[error("Negative Weight: negative weight found at ({row}, {col}) with value {value}")]
    NegativeWeight { row: usize, col: usize, value: f64 },
    #[error("Empty StateSpace: StateSpace must be defined over values")]
    EmptyStateSpace,
}
