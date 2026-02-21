use smallvec::SmallVec;

pub trait Neighborhood {
    fn neighbors(&self, node: usize) -> SmallVec<[usize; 8]>;
    fn num_nodes(&self) -> usize;
}

pub enum Connectivity { Four, Eight }

pub struct Grid2D {
    width: usize,
    height: usize,
    connectivity: Connectivity
}
impl Grid2D {
    pub fn new(width: usize, height: usize, connectivity: Connectivity) -> Self {
        Self { width, height, connectivity}
    }
    #[inline]
    pub fn index(&self, x: usize, y: usize) -> usize {
        y * self.width + x
    }
    
    #[inline]
    pub fn coords(&self, index: usize) -> (usize, usize) {
        (index % self.width, index / self.width)
    }
}
impl Neighborhood for Grid2D {
    fn neighbors(&self, index: usize) -> SmallVec<[usize; 8]> {
        let x = index % self.width;
        let y = index / self.width;
        let mut result = SmallVec::new();
    
        let offsets: &[(isize, isize)] = match self.connectivity {
            Connectivity::Four => &[
                (0, -1), (-1, 0), (1, 0), (0, 1),
            ],
            Connectivity::Eight => &[
                (-1, -1), (0, -1), (1, -1),
                (-1,  0),          (1,  0),
                (-1,  1), (0,  1), (1,  1),
            ],
        };
    
        for &(dx, dy) in offsets {
            let nx = x as isize + dx;
            let ny = y as isize + dy;
            if nx >= 0 && nx < self.width as isize
                && ny >= 0 && ny < self.height as isize
            {
                result.push(ny as usize * self.width + nx as usize);
            }
        }
        result
    }
    #[inline]
    fn num_nodes(&self) -> usize {
        self.width * self.height
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid4(w: usize, h: usize) -> Grid2D {
        Grid2D { width: w, height: h, connectivity: Connectivity::Four }
    }

    fn grid8(w: usize, h: usize) -> Grid2D {
        Grid2D { width: w, height: h, connectivity: Connectivity::Eight }
    }

    // --- index / coords round-trip ---

    #[test]
    fn index_coords_roundtrip() {
        let grid = grid4(5, 5);
        for y in 0..5 {
            for x in 0..5 {
                let idx = grid.index(x, y);
                assert_eq!(grid.coords(idx), (x, y));
            }
        }
    }

    #[test]
    fn index_is_row_major() {
        let grid = grid4(4, 3);
        assert_eq!(grid.index(0, 0), 0);
        assert_eq!(grid.index(3, 0), 3);
        assert_eq!(grid.index(0, 1), 4);
        assert_eq!(grid.index(3, 2), 11);
    }

    // --- num_nodes ---

    #[test]
    fn num_nodes() {
        assert_eq!(grid4(4, 4).num_nodes(), 16);
        assert_eq!(grid4(1, 1).num_nodes(), 1);
        assert_eq!(grid8(10, 20).num_nodes(), 200);
    }

    // --- Four-connected neighbor counts ---

    #[test]
    fn four_corner_has_2_neighbors() {
        let grid = grid4(4, 4);
        assert_eq!(grid.neighbors(grid.index(0, 0)).len(), 2); // top-left
        assert_eq!(grid.neighbors(grid.index(3, 0)).len(), 2); // top-right
        assert_eq!(grid.neighbors(grid.index(0, 3)).len(), 2); // bottom-left
        assert_eq!(grid.neighbors(grid.index(3, 3)).len(), 2); // bottom-right
    }

    #[test]
    fn four_edge_has_3_neighbors() {
        let grid = grid4(4, 4);
        assert_eq!(grid.neighbors(grid.index(1, 0)).len(), 3); // top edge
        assert_eq!(grid.neighbors(grid.index(0, 1)).len(), 3); // left edge
        assert_eq!(grid.neighbors(grid.index(3, 2)).len(), 3); // right edge
        assert_eq!(grid.neighbors(grid.index(2, 3)).len(), 3); // bottom edge
    }

    #[test]
    fn four_interior_has_4_neighbors() {
        let grid = grid4(4, 4);
        assert_eq!(grid.neighbors(grid.index(1, 1)).len(), 4);
        assert_eq!(grid.neighbors(grid.index(2, 2)).len(), 4);
    }

    // --- Eight-connected neighbor counts ---

    #[test]
    fn eight_corner_has_3_neighbors() {
        let grid = grid8(4, 4);
        assert_eq!(grid.neighbors(grid.index(0, 0)).len(), 3);
        assert_eq!(grid.neighbors(grid.index(3, 0)).len(), 3);
        assert_eq!(grid.neighbors(grid.index(0, 3)).len(), 3);
        assert_eq!(grid.neighbors(grid.index(3, 3)).len(), 3);
    }

    #[test]
    fn eight_edge_has_5_neighbors() {
        let grid = grid8(4, 4);
        assert_eq!(grid.neighbors(grid.index(1, 0)).len(), 5);
        assert_eq!(grid.neighbors(grid.index(0, 1)).len(), 5);
    }

    #[test]
    fn eight_interior_has_8_neighbors() {
        let grid = grid8(4, 4);
        assert_eq!(grid.neighbors(grid.index(1, 1)).len(), 8);
        assert_eq!(grid.neighbors(grid.index(2, 2)).len(), 8);
    }

    // --- Correct neighbor values ---

    #[test]
    fn four_interior_neighbors() {
        let grid = grid4(4, 4);
        let mut n: Vec<usize> = grid.neighbors(5).into_vec();
        n.sort();
        // (1,1) neighbors: up=(1,0)=1, left=(0,1)=4, right=(2,1)=6, down=(1,2)=9
        assert_eq!(n, vec![1, 4, 6, 9]);
    }

    #[test]
    fn eight_interior_correct_neighbors() {
        let grid = grid8(4, 4);
        let mut n: Vec<usize> = grid.neighbors(5).into_vec();
        n.sort();
        // (1,1) eight neighbors:
        // (0,0)=0, (1,0)=1, (2,0)=2, (0,1)=4, (2,1)=6, (0,2)=8, (1,2)=9, (2,2)=10
        assert_eq!(n, vec![0, 1, 2, 4, 6, 8, 9, 10]);
    }

    #[test]
    fn four_top_left_corner_neighbors() {
        let grid = grid4(4, 4);
        let mut n: Vec<usize> = grid.neighbors(0).into_vec();
        n.sort();
        // (0,0) neighbors: right=(1,0)=1, down=(0,1)=4
        assert_eq!(n, vec![1, 4]);
    }

    #[test]
    fn eight_top_left_corner_neighbors() {
        let grid = grid8(4, 4);
        let mut n: Vec<usize> = grid.neighbors(0).into_vec();
        n.sort();
        // (0,0) neighbors: right=(1,0)=1, down=(0,1)=4, diag=(1,1)=5
        assert_eq!(n, vec![1, 4, 5]);
    }

    // --- Edge cases ---

    #[test]
    fn single_cell_grid_has_no_neighbors() {
        assert_eq!(grid4(1, 1).neighbors(0).len(), 0);
        assert_eq!(grid8(1, 1).neighbors(0).len(), 0);
    }

    #[test]
    fn single_row() {
        let grid = grid4(5, 1);
        assert_eq!(grid.neighbors(0).len(), 1);  // left end
        assert_eq!(grid.neighbors(2).len(), 2);  // middle
        assert_eq!(grid.neighbors(4).len(), 1);  // right end
    }

    #[test]
    fn single_column() {
        let grid = grid4(1, 5);
        assert_eq!(grid.neighbors(0).len(), 1);  // top
        assert_eq!(grid.neighbors(2).len(), 2);  // middle
        assert_eq!(grid.neighbors(4).len(), 1);  // bottom
    }

    // --- Symmetry: if A is B's neighbor, B is A's neighbor ---

    #[test]
    fn four_neighbor_symmetry() {
        let grid = grid4(5, 5);
        for i in 0..grid.num_nodes() {
            for &j in grid.neighbors(i).iter() {
                assert!(
                    grid.neighbors(j).contains(&i),
                    "node {} lists {} as neighbor, but not vice versa", i, j
                );
            }
        }
    }

    #[test]
    fn eight_neighbor_symmetry() {
        let grid = grid8(5, 5);
        for i in 0..grid.num_nodes() {
            for &j in grid.neighbors(i).iter() {
                assert!(
                    grid.neighbors(j).contains(&i),
                    "node {} lists {} as neighbor, but not vice versa", i, j
                );
            }
        }
    }

    // --- No self-loops ---

    #[test]
    fn no_self_neighbors() {
        let grid = grid8(5, 5);
        for i in 0..grid.num_nodes() {
            assert!(
                !grid.neighbors(i).contains(&i),
                "node {} lists itself as neighbor", i
            );
        }
    }
}