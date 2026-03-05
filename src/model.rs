use std::marker::PhantomData;
use crate::graph::Neighborhood;
use crate::graph::Graph;
use crate::state::Label;
use crate::potentials::CliquePotential;
use crate::error::MrfError;

pub struct Missing;
pub struct Provided;

pub struct MrfBuilderInit;

impl Default for MrfBuilderInit {
    fn default() -> Self { MrfBuilderInit }
}

impl MrfBuilderInit {
    pub fn graph<L: Label>(self, g: Graph<L>) -> MrfBuilder<L, Provided, Missing> {
        MrfBuilder {
            graph: Some(g),
            potentials: Vec::new(),
            _marker: PhantomData,
        }
    }
}

pub struct MrfBuilder<L, HasGraph, HasPotentials> {
    graph: Option<Graph<L>>,
    potentials: Vec<Box<dyn CliquePotential<L>>>,
    _marker: PhantomData<(L, HasGraph, HasPotentials)>,
}

impl Default for MrfBuilder<(), Missing, Missing> {
    fn default() -> Self {
        MrfBuilder {
            graph: None,
            potentials: Vec::new(),
            _marker: PhantomData,
        }
    }
}

impl<L: Label, HasPotentials> MrfBuilder<L, Missing, HasPotentials> {
    pub fn graph(self, g: Graph<L>) -> MrfBuilder<L, Provided, HasPotentials> {
        MrfBuilder {
            graph: Some(g),
            potentials: self.potentials,
            _marker: PhantomData,
        }
    }
}

// first potential transitions Missing -> Provided
impl<L: Label> MrfBuilder<L, Provided, Missing> {
    pub fn potential(self, p: impl CliquePotential<L> + 'static) 
        -> MrfBuilder<L, Provided, Provided> 
    {
        let mut potentials = self.potentials;
        potentials.push(Box::new(p));
        MrfBuilder {
            graph: self.graph,
            potentials,
            _marker: PhantomData,
        }
    }
}

// additional potentials stay Provided
impl<L: Label> MrfBuilder<L, Provided, Provided> {
    pub fn potential(self, p: impl CliquePotential<L> + 'static) 
        -> MrfBuilder<L, Provided, Provided> 
    {
        let mut potentials = self.potentials;
        potentials.push(Box::new(p));
        MrfBuilder {
            graph: self.graph,
            potentials,
            _marker: PhantomData,
        }
    }
}

// build requires all three
impl<L: Label> MrfBuilder<L, Provided, Provided> {
    pub fn build(self) -> Result<MRF<L>, MrfError> {
        Ok(MRF {
            graph: self.graph.unwrap(),
            potentials: self.potentials,
        })
    }
}

pub struct MRF<L> {
    graph: Graph<L>,
    potentials: Vec<Box<dyn CliquePotential<L>>>,
}
 
pub type SweepCallback<L> = Box<dyn FnMut(usize, &[L])>;

pub struct GenerateOptions<L: Label> {
    pub init: Option<Vec<L>>,
    pub seed: Option<u64>,
    pub on_sweep: Option<SweepCallback<L>>
}

impl<L: Label> MRF<L> {
    pub fn builder() -> MrfBuilderInit {
        MrfBuilderInit
    }
    pub fn graph_mut(&mut self) -> &mut Graph<L> {
        &mut self.graph
    }
    pub fn graph(&self) -> &Graph<L> {
        &self.graph
    }
    pub fn potentials(&self) -> &[Box<dyn CliquePotential<L>>] { 
        &self.potentials 
    }

    pub fn num_nodes(&self) -> usize { 
        self.graph.num_nodes() 
    }

    /// Total energy of the current configuration
    pub fn energy(&self) -> f64 {
        let mut total = 0.0;
        for p in &self.potentials {
            let order = p.order();
            for clique in self.graph.cliques_of_order(order) {
                let states: Vec<L> = clique.members().iter()
                    .map(|&i| self.graph.get_node(i).state().clone())
                    .collect();
                total += p.score(&states);
            }
        }
        total
    }

    /// Energy contribution from cliques involving a specific node (with optional testing of different label)
    fn node_energy_inner(&self, node: usize, override_state: Option<&L>) -> f64 {
        let mut total = 0.0;
        for p in &self.potentials {
            self.graph.for_cliques_containing(node, Some(p.order()), |clique| {
                let states: Vec<L> = clique.members().iter()
                    .map(|&i| {
                        if let (true, Some(s)) = (i == node, override_state) { return s.clone(); }
                        self.graph.get_node(i).state().clone()
                    })
                    .collect();
                total += p.score(&states);
            });
        }
        total
    }
    
    /// Current energy of given node
    pub fn node_energy(&self, node: usize) -> f64 {
        self.node_energy_inner(node, None)
    }
    
    /// Used to assess energy of new Label
    pub fn node_energy_with(&self, node: usize, candidate: &L) -> f64 {
        self.node_energy_inner(node, Some(candidate))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Graph;
    use crate::potentials::TablePotential;

    const EPSILON: f64 = 1e-10;

    fn test_graph() -> Graph<usize> {
        let mut g = Graph::new(16);
        for r in 0..4 {
            for c in 0..4 {
                let i = r * 4 + c;
                if c + 1 < 4 { g.add_edge(i, i + 1); }
                if r + 1 < 4 { g.add_edge(i, i + 4); }
            }
        }
        g.detect_cliques();
        g
    }

    fn test_pairwise() -> TablePotential {
        TablePotential::pairwise(&[
            vec![1.0, 0.5, 0.1],
            vec![0.5, 1.0, 0.3],
            vec![0.1, 0.3, 1.0],
        ]).unwrap()
    }

    // log score for a raw weight
    fn log_score(w: f64) -> f64 {
        (w + EPSILON).ln()
    }

    // --- Builder: required fields ---

    #[test]
    fn build_with_graph_and_potential() {
        let mrf = MRF::<usize>::builder()
            .graph(test_graph())
            .potential(test_pairwise())
            .build();
        assert!(mrf.is_ok());
    }

    // --- Builder: multiple potentials ---

    #[test]
    fn build_with_multiple_potentials() {
        let mrf = MRF::<usize>::builder()
            .graph(test_graph())
            .potential(test_pairwise())
            .potential(test_pairwise())
            .build();
        assert!(mrf.is_ok());
        let mrf = mrf.unwrap();
        assert_eq!(mrf.potentials().len(), 2);
    }

    // --- Accessors ---

    #[test]
    fn num_nodes_matches_graph() {
        let mrf = MRF::<usize>::builder()
            .graph(test_graph())
            .potential(test_pairwise())
            .build()
            .unwrap();
        assert_eq!(mrf.num_nodes(), 16);
    }

    // --- Energy ---

    #[test]
    fn uniform_state_energy_is_consistent() {
        let mut g = test_graph();
        for i in 0..g.num_nodes() {
            g.get_node_mut(i).set_state(0);
        }
        let mrf = MRF::<usize>::builder()
            .graph(g)
            .potential(test_pairwise())
            .build()
            .unwrap();
        let e = mrf.energy();
        // All pairs have same labels -> each edge scores log(1.0 + eps)
        // 4x4 grid, 4-connected: 24 edges
        let expected = 24.0 * log_score(1.0);
        assert!((e - expected).abs() < 1e-9);
    }

    #[test]
    fn node_energy_only_counts_local_cliques() {
        let mut g = test_graph();
        for i in 0..g.num_nodes() {
            g.get_node_mut(i).set_state(0);
        }
        let mrf = MRF::<usize>::builder()
            .graph(g)
            .potential(test_pairwise())
            .build()
            .unwrap();

        // Corner node (0) has 2 neighbors -> 2 pairwise cliques
        let e = mrf.node_energy(0);
        assert!((e - 2.0 * log_score(1.0)).abs() < 1e-9);

        // Interior node (5) has 4 neighbors -> 4 pairwise cliques
        let e = mrf.node_energy(5);
        assert!((e - 4.0 * log_score(1.0)).abs() < 1e-9);
    }

    #[test]
    fn energy_changes_with_label() {
        let mut g = test_graph();
        for i in 0..g.num_nodes() {
            g.get_node_mut(i).set_state(0);
        }
        g.get_node_mut(0).set_state(1);

        let mrf = MRF::<usize>::builder()
            .graph(g)
            .potential(test_pairwise())
            .build()
            .unwrap();

        // Node 0 has label 1, its 2 neighbors have label 0
        // score(1, 0) = log(0.5 + eps) per edge
        let e = mrf.node_energy(0);
        assert!((e - 2.0 * log_score(0.5)).abs() < 1e-9);
    }

    #[test]
    fn total_energy_is_sum_over_all_cliques() {
        let mut g = test_graph();
        for i in 0..g.num_nodes() {
            g.get_node_mut(i).set_state(0);
        }
        g.get_node_mut(0).set_state(1);

        let mrf = MRF::<usize>::builder()
            .graph(g)
            .potential(test_pairwise())
            .build()
            .unwrap();

        // 24 edges total. 2 touch node 0 (label 1-0), 22 are 0-0
        let expected = 2.0 * log_score(0.5) + 22.0 * log_score(1.0);
        assert!((mrf.energy() - expected).abs() < 1e-9);
    }

    // --- node_energy_with ---

    #[test]
    fn node_energy_with_matches_after_set() {
        let mut g = test_graph();
        for i in 0..g.num_nodes() {
            g.get_node_mut(i).set_state(0);
        }
        let mrf = MRF::<usize>::builder()
            .graph(g)
            .potential(test_pairwise())
            .build()
            .unwrap();

        // Hypothetical: what if node 0 were label 2?
        let hypothetical = mrf.node_energy_with(0, &2);
        // score(2, 0) = log(0.1 + eps) per edge, 2 neighbors
        assert!((hypothetical - 2.0 * log_score(0.1)).abs() < 1e-9);

        // Graph is unchanged — node 0 still label 0
        let actual = mrf.node_energy(0);
        assert!((actual - 2.0 * log_score(1.0)).abs() < 1e-9);
    }

    // --- Graph clique detection sanity ---

    #[test]
    fn grid_cliques_are_all_pairwise() {
        let g = test_graph();
        for c in g.maximal_cliques() {
            assert_eq!(c.len(), 2);
        }
        assert_eq!(g.maximal_cliques().len(), 24);
    }
}