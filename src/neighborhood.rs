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