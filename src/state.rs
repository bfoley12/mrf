pub trait StateSpace {
    type State: Clone + PartialEq;
    
    fn states(&self) -> &[Self::State];
}