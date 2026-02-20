use crate::neighborhood::Neighborhood;
use crate::state::StateSpace;
pub struct MRF<S, N, U, P>
where 
    S: StateSpace,
    N: Neighborhood,
    U: UnaryPotential<S>,
    P: PairwisePotential<S>,
{
    state_space: S,
    neighborhood: N,
    unary: U,
    pairwise: P,
}