pub trait StateSpace {
    type State: Clone + PartialEq + StateIndex;
    
    fn states(&self) -> &[Self::State];
}

pub trait StateIndex {
    fn as_index(&self) -> usize;
}