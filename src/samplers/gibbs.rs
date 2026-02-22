use crate::samplers::Sampler;
use crate::state::StateSpace;
use crate::neighborhood::Neighborhood;
use crate::potentials::UnaryPotential;
use crate::potentials::PairwisePotential;

use rand::{Rng, RngExt};
use rand::seq::SliceRandom;

pub struct GibbsSampler {
    pub sweeps: usize,
    pub temperature: f64,
}

impl<S: StateSpace> Sampler<S> for GibbsSampler {
    fn sample(
        &self,
        state_space: &S,
        neighborhood: &impl Neighborhood,
        unary: &impl UnaryPotential<S>,
        pairwise: &impl PairwisePotential<S>,
        field: &mut [S::State],
        rng: &mut (impl Rng + RngExt),
    ) {
        let states = state_space.states();
        let num_nodes = neighborhood.num_nodes();
        let mut indices: Vec<usize> = (0..num_nodes).collect();
        let mut log_scores = vec![0.0_f64; states.len()];

        for _sweep in 0..self.sweeps {
            // Shuffle scan order each sweep
            indices.shuffle(rng);

            for &node in &indices {
                let neighbors = neighborhood.neighbors(node);

                // Compute log-score for each candidate label
                for (k, state) in states.iter().enumerate() {
                    let mut score = unary.log_potential(node, state);

                    for &nbr in neighbors.iter() {
                        score += pairwise.log_potential(node, nbr, state, &field[nbr]);
                    }

                    log_scores[k] = score;
                }

                // Sample from conditional distribution
                let max_log = log_scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let weights: Vec<f64> = log_scores.iter()
                    .map(|&s| ((s - max_log) / self.temperature).exp())
                    .collect();
                let total: f64 = weights.iter().sum();

                let mut r = rng.random_range(0.0..total);
                for (k, &w) in weights.iter().enumerate() {
                    r -= w;
                    if r <= 0.0 {
                        field[node] = states[k].clone();
                        break;
                    }
                }
            }
        }
    }
}