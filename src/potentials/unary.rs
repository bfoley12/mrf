use crate::state::{StateSpace, StateIndex};

pub trait UnaryPotential<S: StateSpace> {
    fn log_potential(&self, index: usize, state: &S::State) -> f64;
}

pub struct NoUnary {}

impl<S: StateSpace> UnaryPotential<S> for NoUnary {
    #[inline]
    fn log_potential(&self, _index: usize, _state: &S::State) -> f64 { 0.0 }
}

pub struct UniformUnary {
    log_potentials: Vec<f64>,
}

impl UniformUnary {
    pub fn default(num_potentials: usize) -> Self {
        Self { log_potentials: vec![0.0; num_potentials] }
    }
    
    // TODO!: 
    // - Validate potentials
    // - Add raw_potential construction
    pub fn new(log_potentials: &[f64]) -> Self {
        Self { log_potentials: log_potentials.to_vec() }
    }
}

impl<S: StateSpace> UnaryPotential<S> for UniformUnary {
    #[inline]
    fn log_potential(&self, _index: usize, state: &S::State) -> f64 {
        self.log_potentials[state.as_index()]
    }
}

pub struct SpatialUnary {
    log_potentials: Vec<f64>,
    num_labels: usize
}

impl SpatialUnary {
    // TODO!: Validate
    pub fn new(log_potentials: &[f64], num_labels: usize) -> Self {
        Self { log_potentials: log_potentials.to_vec(), num_labels }
    }
}

impl<S: StateSpace> UnaryPotential<S> for SpatialUnary {
    fn log_potential(&self, index: usize, state: &S::State) -> f64 {
        self.log_potentials[self.num_labels * index + state.as_index()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{DiscreteLabels, Label, StateSpace};

    fn log_pot<U: UnaryPotential<DiscreteLabels>>(u: &U, index: usize, state: &Label) -> f64 {
        u.log_potential(index, state)
    }

    // --- NoUnary ---

    #[test]
    fn no_unary_always_zero() {
        let nu = NoUnary {};
        let labels = DiscreteLabels::new(4);
        let states = labels.states();
        for node in 0..100 {
            for state in states {
                assert_eq!(log_pot(&nu, node, state), 0.0);
            }
        }
    }

    // --- UniformUnary ---

    #[test]
    fn uniform_default_all_zeros() {
        let uu = UniformUnary::default(3);
        let labels = DiscreteLabels::new(3);
        let states = labels.states();
        for state in states {
            assert_eq!(log_pot(&uu, 0, state), 0.0);
        }
    }

    #[test]
    fn uniform_returns_correct_values() {
        let uu = UniformUnary::new(&[-1.0, -2.0, -3.0]);
        let labels = DiscreteLabels::new(3);
        let states = labels.states();
        assert_eq!(log_pot(&uu, 0, &states[0]), -1.0);
        assert_eq!(log_pot(&uu, 0, &states[1]), -2.0);
        assert_eq!(log_pot(&uu, 0, &states[2]), -3.0);
    }

    #[test]
    fn uniform_ignores_node_index() {
        let uu = UniformUnary::new(&[-1.0, -2.0]);
        let labels = DiscreteLabels::new(2);
        let states = labels.states();
        assert_eq!(log_pot(&uu, 0, &states[0]), log_pot(&uu, 999, &states[0]));
        assert_eq!(log_pot(&uu, 0, &states[1]), log_pot(&uu, 42, &states[1]));
    }

    // --- SpatialUnary ---

    fn test_spatial() -> SpatialUnary {
        // 3 nodes, 2 labels
        // node 0: [-1.0, -2.0]
        // node 1: [-3.0, -4.0]
        // node 2: [-5.0, -6.0]
        SpatialUnary::new(&[-1.0, -2.0, -3.0, -4.0, -5.0, -6.0], 2)
    }

    #[test]
    fn spatial_returns_correct_per_node_values() {
        let su = test_spatial();
        let labels = DiscreteLabels::new(2);
        let states = labels.states();

        assert_eq!(log_pot(&su, 0, &states[0]), -1.0);
        assert_eq!(log_pot(&su, 0, &states[1]), -2.0);
        assert_eq!(log_pot(&su, 1, &states[0]), -3.0);
        assert_eq!(log_pot(&su, 1, &states[1]), -4.0);
        assert_eq!(log_pot(&su, 2, &states[0]), -5.0);
        assert_eq!(log_pot(&su, 2, &states[1]), -6.0);
    }

    #[test]
    fn spatial_different_nodes_different_values() {
        let su = test_spatial();
        let labels = DiscreteLabels::new(2);
        let states = labels.states();

        assert_ne!(log_pot(&su, 0, &states[0]), log_pot(&su, 1, &states[0]));
        assert_ne!(log_pot(&su, 1, &states[0]), log_pot(&su, 2, &states[0]));
    }

    // --- NoUnary vs UniformUnary default equivalence ---

    #[test]
    fn no_unary_equivalent_to_uniform_default() {
        let nu = NoUnary {};
        let uu = UniformUnary::default(3);
        let labels = DiscreteLabels::new(3);
        let states = labels.states();

        for node in 0..10 {
            for state in states {
                assert_eq!(log_pot(&nu, node, state), log_pot(&uu, node, state));
            }
        }
    }

    // --- Edge case: single label ---

    #[test]
    fn uniform_single_label() {
        let uu = UniformUnary::new(&[-5.0]);
        let labels = DiscreteLabels::new(1);
        let states = labels.states();
        assert_eq!(log_pot(&uu, 0, &states[0]), -5.0);
    }

    #[test]
    fn spatial_single_label() {
        let su = SpatialUnary::new(&[-1.0, -2.0, -3.0], 1);
        let labels = DiscreteLabels::new(1);
        let states = labels.states();
        assert_eq!(log_pot(&su, 0, &states[0]), -1.0);
        assert_eq!(log_pot(&su, 1, &states[0]), -2.0);
        assert_eq!(log_pot(&su, 2, &states[0]), -3.0);
    }
}