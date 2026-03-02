mod grid;
mod graph;
mod clique;
mod node;

pub use crate::graph::graph::Graph;
pub use crate::graph::grid::Grid2D;

/// Define a neighborhood for each node within a graph
pub trait Neighborhood {
    fn neighbors(&self, node: usize) -> &[usize];
    fn num_nodes(&self) -> usize;
}

/// Convenient for constructing grids
pub trait Connectivity {
    fn offsets(&self) -> &[(isize, isize)];
}
pub struct Four;
pub struct Eight;

impl Connectivity for Four {
    fn offsets(&self) -> &'static [(isize, isize)] {
        &[(0, -1), (-1, 0), (1, 0), (0, 1)]
    }
}

impl Connectivity for Eight {
    fn offsets(&self) -> &'static [(isize, isize)] {
        &[
            (-1, -1), (0, -1), (1, -1),
            (-1,  0),          (1,  0),
            (-1,  1), (0,  1), (1,  1),
        ]
    }
}