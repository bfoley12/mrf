#[derive(Debug)]
pub enum MrfError{
    AsymmetricMatrix { row: usize, col: usize },
    DimensionMismatch { expected: usize, got: usize },
    NegativeWeight { row: usize, col: usize, value: f64 },
    EmptyStateSpace,
}

impl std::error::Error for MrfError {}

// TODO!: Implement informative erros
impl std::fmt::Display for MrfError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::AsymmetricMatrix { row, col } => write!(f, "Asymmetric Matrix Error"),
            Self::DimensionMismatch { expected, got } => write!(f, "Dimension Mismatch Error"),
            Self::NegativeWeight { row, col, value }=> write!(f, "Negative Weight Error"),
            Self::EmptyStateSpace => write!(f, "Empty StateSpace Error"),
        }
    }
}