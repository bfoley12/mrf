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

impl<S, N, U, P, HasState, HasNeighborhood, HasPairwise> MrfBuilder<S, N, U, P, HasState, HasNeighborhood, HasPairwise> {
    pub fn unary<NewU>(self, u: NewU) -> MrfBuilder<S, N, NewU, P, HasState, HasNeighborhood, HasPairwise> {
        MrfBuilder {
            state_space: self.state_space,
            neighborhood: self.neighborhood,
            unary: u,
            pairwise: self.pairwise,
            _marker: PhantomData,
        }
    }
}
impl<S, N, U, P, HasNeighborhood, HasPairwise,> MrfBuilder<S, N, U, P, Missing, HasNeighborhood, HasPairwise> {
    pub fn state_space<NewS: StateSpace>(self, s: NewS) -> MrfBuilder<NewS, N, U, P, Provided, HasNeighborhood, HasPairwise> {
        MrfBuilder {
            state_space: Some(s),
            neighborhood: self.neighborhood,
            unary: self.unary,
            pairwise: self.pairwise,
            _marker: PhantomData,
        }
    }
}

impl<S, N, U, P, HasState, HasPairwise,> MrfBuilder<S, N, U, P, HasState, Missing, HasPairwise> {
    pub fn neighborhood<NewN: Neighborhood>(self, n: NewN) -> MrfBuilder<S, NewN, U, P, HasState, Provided, HasPairwise> {
        MrfBuilder {
            state_space: self.state_space,
            neighborhood: Some(n),
            unary: self.unary,
            pairwise: self.pairwise,
            _marker: PhantomData,
        }
    }
}

impl<S, N, U, P, HasState, HasNeighborhood> MrfBuilder<S, N, U, P, HasState, HasNeighborhood, Missing> {
    pub fn pairwise<NewP>(self, p: NewP) -> MrfBuilder<S, N, U, NewP, HasState, HasNeighborhood, Provided> {
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

impl MRF<(), (), NoUnary, ()> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neighborhood::{Grid2D, Connectivity};
    use crate::state::DiscreteLabels;
    use crate::potentials::{UniformUnary, MatrixPairwise};

    fn test_labels() -> DiscreteLabels {
        DiscreteLabels::new(3)
    }

    fn test_grid() -> Grid2D {
        Grid2D::new(4, 4, Connectivity::Four)
    }

    fn test_pairwise() -> MatrixPairwise {
        MatrixPairwise::new(&vec![
            vec![1.0, 0.5, 0.1],
            vec![0.5, 1.0, 0.3],
            vec![0.1, 0.3, 1.0],
        ]).unwrap()
    }

    fn test_unary() -> UniformUnary {
        UniformUnary::new(&[1.0, 1.0, 1.0])
    }

    // --- Build succeeds with all required fields ---

    #[test]
    fn build_with_all_required() {
        let _mrf = MRF::builder()
            .state_space(test_labels())
            .neighborhood(test_grid())
            .pairwise(test_pairwise())
            .build();
    }

    // --- Order doesn't matter ---

    #[test]
    fn build_order_pairwise_state_neighborhood() {
        let _mrf = MRF::builder()
            .pairwise(test_pairwise())
            .state_space(test_labels())
            .neighborhood(test_grid())
            .build();
    }

    #[test]
    fn build_order_neighborhood_pairwise_state() {
        let _mrf = MRF::builder()
            .neighborhood(test_grid())
            .pairwise(test_pairwise())
            .state_space(test_labels())
            .build();
    }

    // --- Optional unary ---

    #[test]
    fn build_without_unary_defaults_to_no_unary() {
        let _mrf = MRF::builder()
            .state_space(test_labels())
            .neighborhood(test_grid())
            .pairwise(test_pairwise())
            .build();
    }

    #[test]
    fn build_with_unary() {
        let _mrf = MRF::builder()
            .state_space(test_labels())
            .neighborhood(test_grid())
            .unary(test_unary())
            .pairwise(test_pairwise())
            .build();
    }

    #[test]
    fn unary_at_start() {
        let _mrf = MRF::builder()
            .unary(test_unary())
            .state_space(test_labels())
            .neighborhood(test_grid())
            .pairwise(test_pairwise())
            .build();
    }

    #[test]
    fn unary_at_end() {
        let _mrf = MRF::builder()
            .state_space(test_labels())
            .neighborhood(test_grid())
            .pairwise(test_pairwise())
            .unary(test_unary())
            .build();
    }
}