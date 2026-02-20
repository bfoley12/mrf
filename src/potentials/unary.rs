use crate::state::{StateSpace, StateIndex};

pub trait UnaryPotential<S: StateSpace> {
    fn log_potential(&self, index: usize, state: &S::State) -> f64;
}

pub struct UniformUnary {
    log_potentials: Vec<f64>,
}

impl UniformUnary {
    fn default(num_potentials: usize) -> Self {
        Self { log_potentials: vec![0.0; num_potentials] }
    }
}

impl<S: StateSpace> UnaryPotential<S> for UniformUnary {
    fn log_potential(&self, _index: usize, state: &S::State) -> f64 {
        self.log_potentials[state.as_index()]
    }
}

pub struct SpatialUnary {
    log_potentials: Vec<f64>,
    num_labels: usize
}

impl<S: StateSpace> UnaryPotential<S> for SpatialUnary {
    fn log_potential(&self, index: usize, state: &S::State) -> f64 {
        self.log_potentials[self.num_labels * index + state.as_index()]
    }
}