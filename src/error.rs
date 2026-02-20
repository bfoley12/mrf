pub enum MfrError{
    AsymmetricMatrix { row: usize, col: usize },
    DimensionMismatch { expected: usize, got: usize },
    NegativeWeight { row: usize, col: usize, value: f64 },
    EmptyStateSpace,
}

impl std::error::Error for MfrError {}

impl std::fmt::Display for MfrError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            AsymmetricMatrix => write!(f, "Asymmetric Matrix Error"),
            DimensionMismatch => write!(f, "Dimension Mismatch Error"),
            NegativeWeight => write!(f, "Negative Weight Error"),
            EmptyStateSpace => write!(f, "Empty StateSpace Error"),
        }
    }
}