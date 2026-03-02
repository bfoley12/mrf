pub struct Node<T> {
    index: usize,
    observed: T,
    state: T,
    neighbors: Vec<usize>,
}

impl<T> Node<T> {
    pub fn new(index: usize, observed: T, state: T, neighbors: Vec<usize>) -> Self {
        Self { index, observed, state, neighbors }
    }
    
    //fn default(index: usize) -> Self {
    //    Self { index, observed: 0, state: 0, neighbors: Vec::new()}
    //}
    // 
    pub fn add_edge(&mut self, index: usize) {
        match self.neighbors.binary_search(&index) {
            Ok(_) => {}
            Err(pos) => self.neighbors.insert(pos, index),
        }
    }
    
    pub fn neighbors(&self) -> &[usize] {
        &self.neighbors
    }
}