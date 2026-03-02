use std::marker::PhantomData;
use rand::{Rng, RngExt, SeedableRng};
use rand::rngs::StdRng;
use crate::graph::Neighborhood;
use crate::state::StateSpace;
use crate::potentials::{NoUnary, UnaryPotential, PairwisePotential, CompositePairwise};
use crate::samplers::{GibbsSampler, Annealer};
use crate::error::MrfError;

pub struct Missing;
pub struct Provided;

pub struct MrfBuilder<S, N, U, HasState, HasNeighborhood, HasPairwise> {
    state_space: Option<S>,
    neighborhood: Option<N>,
    unary: U,
    pairwise: CompositePairwise<S>,
    _marker: PhantomData<(HasState, HasNeighborhood, HasPairwise)>
}

impl<S, N, U, HasState, HasNeighborhood, HasPairwise> MrfBuilder<S, N, U, HasState, HasNeighborhood, HasPairwise> {
    pub fn unary<NewU>(self, u: NewU) -> MrfBuilder<S, N, NewU, HasState, HasNeighborhood, HasPairwise> {
        MrfBuilder {
            state_space: self.state_space,
            neighborhood: self.neighborhood,
            unary: u,
            pairwise: self.pairwise,
            _marker: PhantomData,
        }
    }
}
impl<S, N, U, HasNeighborhood, HasPairwise,> MrfBuilder<S, N, U, Missing, HasNeighborhood, HasPairwise> {
    pub fn state_space<NewS: StateSpace>(self, s: NewS) -> MrfBuilder<NewS, N, U, Provided, HasNeighborhood, HasPairwise> {
        MrfBuilder {
            state_space: Some(s),
            neighborhood: self.neighborhood,
            unary: self.unary,
            pairwise: CompositePairwise::default(),
            _marker: PhantomData,
        }
    }
}

impl<S, N, U, HasState, HasPairwise,> MrfBuilder<S, N, U, HasState, Missing, HasPairwise> {
    pub fn neighborhood<NewN: Neighborhood>(self, n: NewN) -> MrfBuilder<S, NewN, U, HasState, Provided, HasPairwise> {
        MrfBuilder {
            state_space: self.state_space,
            neighborhood: Some(n),
            unary: self.unary,
            pairwise: self.pairwise,
            _marker: PhantomData,
        }
    }
}

impl<S: StateSpace, N, U, HasNeighborhood> MrfBuilder<S, N, U, Provided, HasNeighborhood, Missing> {
    pub fn pairwise(self, p: impl PairwisePotential<S> + 'static) -> MrfBuilder<S, N, U, Provided, HasNeighborhood, Provided> {
        MrfBuilder {
            state_space: self.state_space,
            neighborhood: self.neighborhood,
            unary: self.unary,
            pairwise: self.pairwise.with(p),
            _marker: PhantomData,
        }
    }
}
impl<S: StateSpace, N, U, HasNeighborhood> MrfBuilder<S, N, U, Provided, HasNeighborhood, Provided> {
    pub fn pairwise(self, p: impl PairwisePotential<S> + 'static) -> MrfBuilder<S, N, U, Provided, HasNeighborhood, Provided> {
        MrfBuilder {
            state_space: self.state_space,
            neighborhood: self.neighborhood,
            unary: self.unary,
            pairwise: self.pairwise.with(p),
            _marker: PhantomData,
        }
    }
}

impl<S, N, U> MrfBuilder<S, N, U, Provided, Provided, Provided>
where
    S: StateSpace,
    N: Neighborhood,
    U: UnaryPotential<S>,
{
    pub fn validate(&self) -> Result<(), MrfError> {
        self.pairwise.validate()?;
        self.unary.validate(self.pairwise.shape())?;
        Ok(())
    }
    pub fn build(self) -> Result<MRF<S, N, U>, MrfError> {
        self.validate()?;
        
        Ok(MRF {
            state_space: self.state_space.unwrap(),
            neighborhood: self.neighborhood.unwrap(),
            unary: self.unary,
            pairwise: self.pairwise,
        })
    }
}

pub struct MRF<S, N, U> {
    state_space: S,
    neighborhood: N,
    unary: U,
    pairwise: CompositePairwise<S>,
    // TODO!: Add Adjacency so we can generalize to a Graph instead of a Grid
}

impl MRF<(), (), NoUnary> {
    pub fn builder() -> MrfBuilder<(), (), NoUnary, Missing, Missing, Missing> {
        MrfBuilder {
            state_space: None,
            neighborhood: None,
            unary: NoUnary{},
            pairwise: CompositePairwise::default(),
            _marker: PhantomData,
        }
    }
}

pub type SweepCallback<S> = Box<dyn FnMut(usize, &[<S as StateSpace>::State])>;

pub struct GenerateOptions<S: StateSpace> {
    pub init: Option<Vec<S::State>>,
    pub seed: Option<u64>,
    pub on_sweep: Option<SweepCallback<S>>
}

impl<S, N, U> MRF<S, N, U> 
where
    S: StateSpace,
    N: Neighborhood,
    U: UnaryPotential<S>,
{
    pub fn random_init(&self, rng: &mut impl Rng) -> Vec<S::State> {
        let states = self.state_space.states();
        (0..self.neighborhood.num_nodes())
            .map(|_| states[rng.random_range(0..states.len())].clone())
            .collect()
    }

    pub fn generate<A: Annealer, R: RngExt>(
        &self,
        sampler: &GibbsSampler<A>,
        mut opts: GenerateOptions<S>,
    ) -> Result<Vec<S::State>, MrfError> {
        let mut rng = match opts.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_rng(&mut rand::rng()),
        };
        let mut field = match opts.init {
            Some(f) => {
                if f.len() != self.neighborhood.num_nodes() {
                    return Err(MrfError::DimensionMismatch {
                        expected: self.neighborhood.num_nodes(),
                        got: f.len(),
                    });
                }
                f
            }
            None => self.random_init(&mut rng),
        };
        if let Some(cb) = &mut opts.on_sweep { cb(0, &field); }
        for i in 0..sampler.sweeps() {
            let temp = sampler.annealer().temperature(i);
            sampler.sweep(temp, &self.state_space, &self.neighborhood, &self.unary, &self.pairwise, &mut field, &mut rng);
            if let Some(cb) = &mut opts.on_sweep { cb(i + 1, &field); }
        }
        Ok(field)
    }
    pub fn state_space(&self) -> &S { &self.state_space }
    pub fn neighborhood(&self) -> &N { &self.neighborhood }
    pub fn unary(&self) -> &U { &self.unary }
    pub fn pairwise(&self) -> &CompositePairwise<S> { &self.pairwise }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Grid2D, Four};
    use crate::state::DiscreteLabels;
    use crate::potentials::{UniformUnary, MatrixPairwise};

    fn test_labels() -> DiscreteLabels {
        DiscreteLabels::new(3)
    }

    fn test_grid() -> Grid2D<()> {
        Grid2D::new(4, 4, Four)
    }

    fn test_pairwise() -> MatrixPairwise {
        MatrixPairwise::new(&[
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
    fn build_order_state_pairwise_neighborhood() {
        let _mrf = MRF::builder()
            .state_space(test_labels())
            .pairwise(test_pairwise())
            .neighborhood(test_grid())
            .build();
    }

    #[test]
    fn build_order_neighborhood_state_pairwise() {
        let _mrf = MRF::builder()
            .neighborhood(test_grid())
            .state_space(test_labels())
            .pairwise(test_pairwise())
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
    
    #[test]
    fn multiple_pairwise() {
        let _mrf = MRF::builder()
            .state_space(test_labels())
            .neighborhood(test_grid())
            .pairwise(test_pairwise())
            .unary(test_unary())
            .pairwise(test_pairwise())
            .build();
    }
}