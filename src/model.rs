use crate::neighborhood::Neighborhood;
use crate::state::StateSpace;
use crate::potentials::{UnaryPotential, PairwisePotential}
pub struct MRF<S, N, U, P>
where 
    S: StateSpace,
    N: Neighborhood,
    U: UnaryPotential<S>,
    P: PairwisePotential<S>,
{
    state_space: S,
    neighborhood: N,
    unary: Option<U>,
    pairwise: P,
    // Later can add adjacency layer for general Graphs
}