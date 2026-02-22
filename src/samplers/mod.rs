use crate::state::StateSpace;
use crate::neighborhood::Neighborhood;
use crate::potentials::UnaryPotential;
use crate::potentials::PairwisePotential;

use rand::Rng;

mod gibbs;
pub use self::gibbs::GibbsSampler;

pub trait Sampler<S: StateSpace> {
    fn sample(
        &self,
        state_space: &S,
        neighborhood: &impl Neighborhood,
        unary: &impl UnaryPotential<S>,
        pairwise: &impl PairwisePotential<S>,
        field: &mut [S::State],
        rng: &mut impl Rng,
    );
}