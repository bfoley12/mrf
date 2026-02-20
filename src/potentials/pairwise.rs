use crate::state::{StateSpace, StateIndex};
use crate::error::MrfError;

pub trait PairwisePotential<S: StateSpace> {
    fn log_potential(&self, i: usize, j: usize, i_state: &S::State, j_state: &S::State) -> f64;
    fn num_labels(&self) -> usize;
}

pub struct MatrixPairwise {
    log_potentials: Vec<f64>,
    num_labels: usize,
}

impl<S: StateSpace> PairwisePotential<S> for MatrixPairwise {
    #[inline]
    fn log_potential(&self, i: usize, j: usize, i_state: &S::State, j_state: &S::State) -> f64 {
        self.log_potentials[i_state.as_index() * self.num_labels + j_state.as_index()]
    }
    #[inline]
    fn num_labels(&self) -> usize {
        self.num_labels
    }
}

impl MatrixPairwise {
    pub fn new(raw_weights: &[Vec<f64>]) -> Result<Self, MrfError> {
        let n = raw_weights.len();
        if n == 0 {
            return Err(MrfError::EmptyStateSpace);
        }
    
        let epsilon = 1e-10;
        let mut log_potentials = vec![0.0_f64; n * n];
    
        for i in 0..n {
            if raw_weights[i].len() != n {
                return Err(MrfError::DimensionMismatch {
                    expected: n,
                    got: raw_weights[i].len(),
                });
            }
    
            for j in i..n {
                let val = raw_weights[i][j];
    
                if val < 0.0 {
                    return Err(MrfError::NegativeWeight { row: i, col: j, value: val });
                }
    
                if j > i && (val - raw_weights[j][i]).abs() > epsilon {
                    return Err(MrfError::AsymmetricMatrix { row: i, col: j });
                }
    
                let log_val = (val + epsilon).ln();
                log_potentials[i * n + j] = log_val;
                log_potentials[j * n + i] = log_val;
            }
        }
    
        Ok(Self { log_potentials, num_labels: n })
    }
}