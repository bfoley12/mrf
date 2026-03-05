use crate::MRF;
use crate::error::MrfError;
use crate::state::Label;
use rand::Rng;

mod gibbs;
mod annealers;
pub use self::annealers::{ConstantAnnealer, LinearAnnealer, ExponentialAnnealer, LogarithmicAnnealer};
pub use self::gibbs::{GibbsSampler, RunOptions};

pub trait Sampler<L: Label> {
    fn run(
        &self,
        mrf: &mut MRF<L>,
        proposal: &impl Proposal<L>,
        opts: RunOptions,
    ) -> Result<(), MrfError>;
}

pub trait Annealer {
    fn temperature(&self, sweep: usize) -> f64;
}

pub trait Proposal<L: Label> {
    fn candidates(&self, current: &L, rng: &mut impl Rng) -> Vec<L>;
}

/// Enumerates all discrete labels — use with usize, Label(usize), etc.
pub struct DiscreteProposal {
    num_labels: usize,
}

impl DiscreteProposal {
    pub fn new(num_labels: usize) -> Self {
        Self { num_labels }
    }
}
impl Proposal<usize> for DiscreteProposal {
    fn candidates(&self, _current: &usize, _rng: &mut impl Rng) -> Vec<usize> {
        (0..self.num_labels).collect()
    }
}