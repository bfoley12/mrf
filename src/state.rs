pub trait StateSpace {
    type State: Clone + PartialEq + StateIndex;
    
    fn states(&self) -> &[Self::State];
}

pub trait StateIndex {
    fn as_index(&self) -> usize;
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct Label(pub usize);

impl StateIndex for Label {
    #[inline]
    fn as_index(&self) -> usize {
        self.0
    }
}
pub struct DiscreteLabels {
    labels: Vec<Label>,
}

impl DiscreteLabels {
    pub fn new(num_labels: usize) -> Self {
        Self { labels: (0..num_labels).map(Label).collect()}
    }
}

impl StateSpace for DiscreteLabels {
    type State = Label;
    
    #[inline]
    fn states(&self) -> &[Self::State] {
        &self.labels
    }
}