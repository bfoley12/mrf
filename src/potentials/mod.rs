mod table;

pub use table::*;

use crate::state::Label;

pub trait HasShape {
    fn shape(&self) -> (usize, usize);
}

pub trait CliquePotential<L: Label>: Send + Sync {
    fn order(&self) -> usize;
    fn score(&self, states: &[L]) -> f64;
}