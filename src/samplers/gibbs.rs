use crate::state::Label;
use crate::MRF;
use crate::samplers::Annealer;
use crate::error::MrfError;
use rand::{Rng, RngExt};
use rand::rngs::StdRng;
use rand::{SeedableRng};
use rand::seq::SliceRandom;
use crate::samplers::{Proposal};

#[derive(Default)]
pub struct RunOptions {
    pub seed: Option<u64>,
}

pub struct GibbsSampler<A: Annealer> {
    sweeps: usize,
    annealer: A,
}

impl<A: Annealer> GibbsSampler<A> {
    pub fn new(sweeps: usize, annealer: A) -> Self {
        Self { sweeps, annealer }
    }

    pub fn sweeps(&self) -> usize {
        self.sweeps
    }

    pub fn annealer(&self) -> &A {
        &self.annealer
    }

    fn sweep<L: Label>(
        &self,
        temperature: f64,
        mrf: &mut MRF<L>,
        proposal: &impl Proposal<L>,
        rng: &mut impl Rng,
    ) {
        let num_nodes = mrf.num_nodes();
        let mut indices: Vec<usize> = (0..num_nodes).collect();
        indices.shuffle(rng);

        for &node in &indices {
            let current = mrf.graph().get_node(node).state().clone();
            let candidates = proposal.candidates(&current, rng);
            let mut log_scores: Vec<f64> = Vec::with_capacity(candidates.len());

            for candidate in &candidates {
                // Temporarily set candidate to compute energy
                mrf.graph_mut().get_node_mut(node).set_state(candidate.clone());
                log_scores.push(-mrf.node_energy(node));
            }

            // Sample from conditional via Gibbs
            let max_log = log_scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let weights: Vec<f64> = log_scores.iter()
                .map(|&s| ((s - max_log) / temperature).exp())
                .collect();
            let total: f64 = weights.iter().sum();
            let mut r = rng.random_range(0.0..total);

            for (k, &w) in weights.iter().enumerate() {
                r -= w;
                if r <= 0.0 {
                    mrf.graph_mut().get_node_mut(node).set_state(candidates[k].clone());
                    break;
                }
            }
        }
    }

    pub fn run<L: Label>(
        &self,
        mrf: &mut MRF<L>,
        proposal: &impl Proposal<L>,
        opts: RunOptions,
    ) -> Result<(), MrfError> {
        let mut rng = match opts.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_rng(&mut rand::rng()),
        };
        for i in 0..self.sweeps {
            let temp = self.annealer.temperature(i);
            self.sweep(temp, mrf, proposal, &mut rng);
        }
        Ok(())
    }
    
    pub fn run_with<L: Label>(
        &self,
        mrf: &mut MRF<L>,
        proposal: &impl Proposal<L>,
        opts: RunOptions,
        mut on_sweep: impl FnMut(usize, &MRF<L>),
    ) -> Result<(), MrfError> {
        let mut rng = match opts.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_rng(&mut rand::rng()),
        };
        on_sweep(0, mrf);
        for i in 0..self.sweeps {
            let temp = self.annealer.temperature(i);
            self.sweep(temp, mrf, proposal, &mut rng);
            on_sweep(i + 1, mrf);
        }
        Ok(())
    }
}