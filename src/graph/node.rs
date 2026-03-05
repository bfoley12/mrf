pub struct Node<T> {
    state: T,
    neighbors: Vec<usize>,
}

impl<T> Node<T> {
    pub fn new(state: T, neighbors: Vec<usize>) -> Self {
        Self { state, neighbors }
    }

    pub fn add_edge(&mut self, index: usize) {
        match self.neighbors.binary_search(&index) {
            Ok(_) => {}
            Err(pos) => self.neighbors.insert(pos, index),
        }
    }
    
    pub fn neighbors(&self) -> &[usize] {
        &self.neighbors
    }
    pub fn state(&self) -> &T {
        &self.state
    }
    pub fn set_state(&mut self, state: T) {
        self.state = state;
    }
}