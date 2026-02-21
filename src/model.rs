use std::marker::PhantomData;
use crate::neighborhood::Neighborhood;
use crate::state::StateSpace;
use crate::potentials::{NoUnary, UnaryPotential, PairwisePotential};

pub struct Missing;
pub struct Provided;

pub struct MrfBuilder<S, N, U, P, HasState, HasNeighborhood, HasPairwise> {
    state_space: Option<S>,
    neighborhood: Option<N>,
    unary: U,
    pairwise: Option<P>,
    // sampler
    // temperature
    // sweeps?

    _marker: PhantomData<(HasState, HasNeighborhood, HasPairwise)>
}

impl<S, N, U, P, HasState, HasNeighborhood, HasPairwise> MrfBuilder<S, N, U, P, HasState, HasNeighborhood, HasPairwise>
where
    S: StateSpace,
    N: Neighborhood,
    U: UnaryPotential<S>,
    P: PairwisePotential<S>,
{
    pub fn unary<NewU: UnaryPotential<S>>(self, u: NewU) -> MrfBuilder<S, N, NewU, P, HasState, HasNeighborhood, HasPairwise> {
        MrfBuilder {
            state_space: self.state_space,
            neighborhood: self.neighborhood,
            unary: u,
            pairwise: self.pairwise,
            _marker: PhantomData,
        }
    }
}
impl<S, N, U, P, HasNeighborhood, HasPairwise,> MrfBuilder<S, N, U, P, Missing, HasNeighborhood, HasPairwise>
where
    S: StateSpace,
    N: Neighborhood,
    U: UnaryPotential<S>,
    P: PairwisePotential<S>,
{
    pub fn state_space(self, s: S) -> MrfBuilder<S, N, U, P, Provided, HasNeighborhood, HasPairwise> {
        MrfBuilder {
            state_space: Some(s),
            neighborhood: self.neighborhood,
            unary: self.unary,
            pairwise: self.pairwise,
            _marker: PhantomData,
        }
    }
}

impl<S, N, U, P, HasState, HasPairwise,> MrfBuilder<S, N, U, P, HasState, Missing, HasPairwise>
where
    S: StateSpace,
    N: Neighborhood,
    U: UnaryPotential<S>,
    P: PairwisePotential<S>,
{
    pub fn neighborhood(self, n: N) -> MrfBuilder<S, N, U, P, HasState, Provided, HasPairwise> {
        MrfBuilder {
            state_space: self.state_space,
            neighborhood: Some(n),
            unary: self.unary,
            pairwise: self.pairwise,
            _marker: PhantomData,
        }
    }
}

impl<S, N, U, P, HasState, HasNeighborhood> MrfBuilder<S, N, U, P, HasState, HasNeighborhood, Missing>
where
    S: StateSpace,
    N: Neighborhood,
    U: UnaryPotential<S>,
    P: PairwisePotential<S>,
{
    pub fn pairwise(self, p: P) -> MrfBuilder<S, N, U, P, HasState, HasNeighborhood, Provided> {
        MrfBuilder {
            state_space: self.state_space,
            neighborhood: self.neighborhood,
            unary: self.unary,
            pairwise: Some(p),
            _marker: PhantomData,
        }
    }
}

impl<S, N, U, P> MrfBuilder<S, N, U, P, Provided, Provided, Provided>
where
    S: StateSpace,
    N: Neighborhood,
    U: UnaryPotential<S>,
    P: PairwisePotential<S>,
{
    pub fn build(self) -> MRF<S, N, U, P> {
        MRF {
            state_space: self.state_space.unwrap(),
            neighborhood: self.neighborhood.unwrap(),
            unary: self.unary,
            pairwise: self.pairwise.unwrap(),
        }
    }
}

pub struct MRF<S, N, U, P> {
    state_space: S,
    neighborhood: N,
    unary: U,
    pairwise: P,
    // Later can add adjacency layer for general Graphs
    // TODO!:
    // - Temperature (as field) and annealing
    // - sampler (add as field and as struct in samplers folder)
    // - look into Unary as Box<dyn 
}

impl<S, N, U, P> MRF<S, N, U, P> 
where
    S: StateSpace,
    N: Neighborhood,
    U: UnaryPotential<S>,
    P: PairwisePotential<S>,
{
    pub fn builder() -> MrfBuilder<(), (), NoUnary, (), Missing, Missing, Missing> {
        MrfBuilder {
            state_space: None,
            neighborhood: None,
            unary: NoUnary{},
            pairwise: None,
            _marker: PhantomData,
        }
    }
}