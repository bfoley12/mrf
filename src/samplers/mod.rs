use crate::state::StateSpace;
use crate::neighborhood::Neighborhood;
use crate::potentials::UnaryPotential;
use crate::potentials::PairwisePotential;

use rand::{Rng, RngExt};

mod gibbs;
pub use self::gibbs::GibbsSampler;

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