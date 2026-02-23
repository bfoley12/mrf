mod unary;
mod pairwise;

pub use unary::*;
pub use pairwise::*;

use crate::state::StateSpace;
use crate::error::MrfError;

pub trait HasShape {
    fn shape(&self) -> (usize, usize);
}

pub trait PairwisePotential<S: StateSpace>: HasShape {
    fn log_potential(&self, i: usize, j: usize, i_state: &S::State, j_state: &S::State) -> f64;
    fn num_labels(&self) -> usize;
}

pub trait UnaryPotential<S: StateSpace>: HasShape {
    fn log_potential(&self, index: usize, state: &S::State) -> f64;
    fn validate(&self, valid_shape: (usize, usize)) -> Result<(), MrfError>;
}