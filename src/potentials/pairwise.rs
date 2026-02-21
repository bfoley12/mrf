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
    fn log_potential(&self, _i: usize, _j: usize, i_state: &S::State, j_state: &S::State) -> f64 {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{DiscreteLabels, Label, StateSpace};

    // --- Construction: valid inputs ---

    #[test]
    fn new_symmetric_matrix() {
        let result = MatrixPairwise::new(&vec![
            vec![1.0, 0.5],
            vec![0.5, 1.0],
        ]);
        assert!(result.is_ok());
    }

    #[test]
    fn new_single_label() {
        let result = MatrixPairwise::new(&vec![vec![1.0]]);
        assert!(result.is_ok());
    }

    #[test]
    fn new_with_zeros() {
        let result = MatrixPairwise::new(&vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
        ]);
        assert!(result.is_ok());
    }

    #[test]
    fn num_labels_matches() {
        let mp = MatrixPairwise::new(&vec![
            vec![1.0, 0.5, 0.1],
            vec![0.5, 1.0, 0.3],
            vec![0.1, 0.3, 1.0],
        ]).unwrap();
        assert_eq!(mp.num_labels, 3);
    }

    // --- Construction: invalid inputs ---

    #[test]
    fn empty_matrix_fails() {
        let result = MatrixPairwise::new(&vec![]);
        assert!(matches!(result, Err(MrfError::EmptyStateSpace)));
    }

    #[test]
    fn non_square_matrix_fails() {
        let result = MatrixPairwise::new(&vec![
            vec![1.0, 0.5, 0.1],
            vec![0.5, 1.0],
        ]);
        assert!(matches!(result, Err(MrfError::DimensionMismatch { expected: 2, got: 3 })
            | Err(MrfError::DimensionMismatch { expected: 3, got: 2 })));
    }

    #[test]
    fn negative_weight_fails() {
        let result = MatrixPairwise::new(&vec![
            vec![1.0, -0.5],
            vec![-0.5, 1.0],
        ]);
        assert!(matches!(result, Err(MrfError::NegativeWeight { .. })));
    }

    #[test]
    fn asymmetric_matrix_fails() {
        let result = MatrixPairwise::new(&vec![
            vec![1.0, 0.8],
            vec![0.3, 1.0],
        ]);
        assert!(matches!(result, Err(MrfError::AsymmetricMatrix { row: 0, col: 1 })));
    }

    #[test]
    fn asymmetry_within_epsilon_passes() {
        let tiny = 1e-12;
        let result = MatrixPairwise::new(&vec![
            vec![1.0, 0.5 + tiny],
            vec![0.5, 1.0],
        ]);
        assert!(result.is_ok());
    }

    #[test]
    fn asymmetry_beyond_epsilon_fails() {
        let result = MatrixPairwise::new(&vec![
            vec![1.0, 0.5 + 1e-8],
            vec![0.5, 1.0],
        ]);
        assert!(matches!(result, Err(MrfError::AsymmetricMatrix { .. })));
    }

    // --- Log potential values ---

    #[test]
    fn log_potential_is_ln_of_input_plus_epsilon() {
        let mp = MatrixPairwise::new(&vec![
            vec![1.0, 0.5],
            vec![0.5, 1.0],
        ]).unwrap();

        let labels = DiscreteLabels::new(2);
        let states = labels.states();
        let epsilon = 1e-10;

        let val = <MatrixPairwise as PairwisePotential<DiscreteLabels>>::log_potential(
            &mp, 0, 1, &states[0], &states[0],
        );
        assert!((val - (1.0_f64 + epsilon).ln()).abs() < 1e-12);

        let val = <MatrixPairwise as PairwisePotential<DiscreteLabels>>::log_potential(
            &mp, 0, 1, &states[0], &states[1],
        );
        assert!((val - (0.5_f64 + epsilon).ln()).abs() < 1e-12);
    }

    #[test]
    fn log_potential_symmetric() {
        let mp = MatrixPairwise::new(&vec![
            vec![1.0, 0.3],
            vec![0.3, 1.0],
        ]).unwrap();

        let labels = DiscreteLabels::new(2);
        let states = labels.states();

        let ab = <MatrixPairwise as PairwisePotential<DiscreteLabels>>::log_potential(
            &mp, 0, 1, &states[0], &states[1],
        );
        let ba = <MatrixPairwise as PairwisePotential<DiscreteLabels>>::log_potential(
            &mp, 0, 1, &states[1], &states[0],
        );
        assert!((ab - ba).abs() < 1e-12);
    }

    #[test]
    fn log_potential_ignores_node_indices() {
        let mp = MatrixPairwise::new(&vec![
            vec![1.0, 0.5],
            vec![0.5, 1.0],
        ]).unwrap();

        let labels = DiscreteLabels::new(2);
        let states = labels.states();

        let a = <MatrixPairwise as PairwisePotential<DiscreteLabels>>::log_potential(
            &mp, 0, 1, &states[0], &states[1],
        );
        let b = <MatrixPairwise as PairwisePotential<DiscreteLabels>>::log_potential(
            &mp, 99, 1000, &states[0], &states[1],
        );
        assert_eq!(a, b);
    }

    #[test]
    fn zero_weight_produces_very_negative_log() {
        let mp = MatrixPairwise::new(&vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
        ]).unwrap();

        let labels = DiscreteLabels::new(2);
        let states = labels.states();

        let val = <MatrixPairwise as PairwisePotential<DiscreteLabels>>::log_potential(
            &mp, 0, 1, &states[0], &states[1],
        );
        // ln(0 + 1e-10) ≈ -23.03
        assert!(val < -20.0);
    }

    // --- Diagonal vs off-diagonal ---

    #[test]
    fn diagonal_higher_than_off_diagonal() {
        let mp = MatrixPairwise::new(&vec![
            vec![1.0, 0.1],
            vec![0.1, 1.0],
        ]).unwrap();

        let labels = DiscreteLabels::new(2);
        let states = labels.states();

        let same = <MatrixPairwise as PairwisePotential<DiscreteLabels>>::log_potential(
            &mp, 0, 1, &states[0], &states[0],
        );
        let diff = <MatrixPairwise as PairwisePotential<DiscreteLabels>>::log_potential(
            &mp, 0, 1, &states[0], &states[1],
        );
        assert!(same > diff);
    }
}