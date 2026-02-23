use crate::state::StateSpace;
use crate::neighborhood::Neighborhood;
use crate::potentials::UnaryPotential;
use crate::potentials::PairwisePotential;

use rand::{Rng, RngExt};

mod gibbs;
mod annealers;
pub use self::gibbs::GibbsSampler;
pub use self::annealers::{ConstantAnnealer, LinearAnnealer, ExponentialAnnealer, LogarithmicAnnealer};

pub trait Sampler<S: StateSpace> {
    fn sample<N, U, P, R> (
        &self,
        state_space: &S,
        neighborhood: &N,
        unary: &U,
        pairwise: &P,
        field: &mut [S::State],
        rng: &mut R,
    )
    where 
        N: Neighborhood,
        U: UnaryPotential<S>,
        P: PairwisePotential<S>,
        R: Rng + RngExt
    ;
}

pub trait Annealer {
    fn temperature(&self, sweep: usize) -> f64;
}

