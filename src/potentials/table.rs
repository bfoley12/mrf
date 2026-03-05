use crate::potentials::CliquePotential;
use crate::error::MrfError;

/// A lookup-table potential for cliques of any order.
/// Stores scores in a flattened N^order array indexed by label combinations.
pub struct TablePotential {
    scores: Vec<f64>,
    num_labels: usize,
    order: usize,
}

impl TablePotential {
    /// Build from a flat vec of scores.
    /// Length must be num_labels^order.
    /// Index convention: scores[l0 * N^(k-1) + l1 * N^(k-2) + ... + l_{k-1}]
    pub fn new(scores: Vec<f64>, num_labels: usize, order: usize) -> Result<Self, MrfError> {
        let expected = num_labels.pow(order as u32);
        if scores.len() != expected {
            return Err(MrfError::DimensionMismatch {
                expected,
                got: scores.len(),
            });
        }
        Ok(Self { scores, num_labels, order })
    }

    /// Build a pairwise potential from a weight matrix.
    /// Validates symmetry and non-negativity, stores as log scores.
    pub fn pairwise(weights: &[Vec<f64>]) -> Result<Self, MrfError> {
        let n = weights.len();
        if n == 0 {
            return Err(MrfError::EmptyStateSpace);
        }

        let epsilon = 1e-10;
        let mut scores = vec![0.0_f64; n * n];

        for i in 0..n {
            if weights[i].len() != n {
                return Err(MrfError::DimensionMismatch {
                    expected: n,
                    got: weights[i].len(),
                });
            }
            for j in i..n {
                let val = weights[i][j];
                if val < 0.0 {
                    return Err(MrfError::NegativeWeight { row: i, col: j, value: val });
                }
                if j > i && (val - weights[j][i]).abs() > epsilon {
                    return Err(MrfError::AsymmetricMatrix { row: i, col: j });
                }
                let log_val = (val + epsilon).ln();
                scores[i * n + j] = log_val;
                scores[j * n + i] = log_val;
            }
        }

        Ok(Self { scores, num_labels: n, order: 2 })
    }

    /// Build a unary potential from per-label scores.
    pub fn unary(scores: Vec<f64>) -> Self {
        Self {
            num_labels: scores.len(),
            order: 1,
            scores,
        }
    }

    pub fn num_labels(&self) -> usize { self.num_labels }
    pub fn order(&self) -> usize { self.order }

    fn flat_index(&self, states: &[usize]) -> usize {
        let mut idx = 0;
        for &s in states {
            idx = idx * self.num_labels + s;
        }
        idx
    }
}

impl CliquePotential<usize> for TablePotential {
    fn order(&self) -> usize { self.order }

    fn score(&self, states: &[usize]) -> f64 {
        debug_assert_eq!(states.len(), self.order);
        self.scores[self.flat_index(states)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-10;

    // --- Pairwise construction ---

    #[test]
    fn pairwise_valid() {
        let tp = TablePotential::pairwise(&[
            vec![1.0, 0.5],
            vec![0.5, 1.0],
        ]);
        assert!(tp.is_ok());
        let tp = tp.unwrap();
        assert_eq!(tp.order(), 2);
        assert_eq!(tp.num_labels(), 2);
    }

    #[test]
    fn pairwise_empty_fails() {
        assert!(matches!(
            TablePotential::pairwise(&[]),
            Err(MrfError::EmptyStateSpace)
        ));
    }

    #[test]
    fn pairwise_non_square_fails() {
        assert!(matches!(
            TablePotential::pairwise(&[vec![1.0, 0.5, 0.1], vec![0.5, 1.0]]),
            Err(MrfError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn pairwise_negative_fails() {
        assert!(matches!(
            TablePotential::pairwise(&[vec![1.0, -0.5], vec![-0.5, 1.0]]),
            Err(MrfError::NegativeWeight { .. })
        ));
    }

    #[test]
    fn pairwise_asymmetric_fails() {
        assert!(matches!(
            TablePotential::pairwise(&[vec![1.0, 0.8], vec![0.3, 1.0]]),
            Err(MrfError::AsymmetricMatrix { row: 0, col: 1 })
        ));
    }

    // --- Pairwise scores ---

    #[test]
    fn pairwise_score_values() {
        let tp = TablePotential::pairwise(&[
            vec![1.0, 0.5],
            vec![0.5, 1.0],
        ]).unwrap();

        assert!((tp.score(&[0, 0]) - (1.0_f64 + EPSILON).ln()).abs() < 1e-12);
        assert!((tp.score(&[0, 1]) - (0.5_f64 + EPSILON).ln()).abs() < 1e-12);
    }

    #[test]
    fn pairwise_score_symmetric() {
        let tp = TablePotential::pairwise(&[
            vec![1.0, 0.3],
            vec![0.3, 1.0],
        ]).unwrap();
        assert!((tp.score(&[0, 1]) - tp.score(&[1, 0])).abs() < 1e-12);
    }

    #[test]
    fn pairwise_diagonal_preferred() {
        let tp = TablePotential::pairwise(&[
            vec![1.0, 0.1],
            vec![0.1, 1.0],
        ]).unwrap();
        assert!(tp.score(&[0, 0]) > tp.score(&[0, 1]));
    }

    // --- Unary ---

    #[test]
    fn unary_construction() {
        let tp = TablePotential::unary(vec![0.5, 1.0, 0.2]);
        assert_eq!(tp.order(), 1);
        assert_eq!(tp.num_labels(), 3);
    }

    #[test]
    fn unary_scores() {
        let tp = TablePotential::unary(vec![0.5, 1.0, 0.2]);
        assert!((tp.score(&[0]) - 0.5).abs() < 1e-12);
        assert!((tp.score(&[1]) - 1.0).abs() < 1e-12);
        assert!((tp.score(&[2]) - 0.2).abs() < 1e-12);
    }

    // --- Generic construction ---

    #[test]
    fn new_valid_dimensions() {
        // 3 labels, order 2 -> 9 entries
        let tp = TablePotential::new(vec![0.0; 9], 3, 2);
        assert!(tp.is_ok());
    }

    #[test]
    fn new_wrong_length_fails() {
        let tp = TablePotential::new(vec![0.0; 10], 3, 2);
        assert!(matches!(tp, Err(MrfError::DimensionMismatch { expected: 9, got: 10 })));
    }

    #[test]
    fn new_higher_order() {
        // 2 labels, order 3 -> 8 entries
        let scores: Vec<f64> = (0..8).map(|i| i as f64).collect();
        let tp = TablePotential::new(scores, 2, 3).unwrap();
        assert_eq!(tp.order(), 3);
        // [0,0,0] -> index 0, [1,1,0] -> index 6
        assert!((tp.score(&[0, 0, 0]) - 0.0).abs() < 1e-12);
        assert!((tp.score(&[1, 1, 0]) - 6.0).abs() < 1e-12);
    }

    // --- Flat index ---

    #[test]
    fn flat_index_pairwise() {
        let tp = TablePotential::new(vec![0.0; 9], 3, 2).unwrap();
        // [1, 2] -> 1*3 + 2 = 5
        assert_eq!(tp.flat_index(&[1, 2]), 5);
    }

    #[test]
    fn flat_index_ternary() {
        let tp = TablePotential::new(vec![0.0; 8], 2, 3).unwrap();
        // [1, 0, 1] -> 1*4 + 0*2 + 1 = 5
        assert_eq!(tp.flat_index(&[1, 0, 1]), 5);
    }
}