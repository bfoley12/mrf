use crate::graph::Neighborhood;
use crate::graph::node::Node;
use crate::graph::clique::Clique;

pub struct Graph<T> {
    nodes: Vec<Node<T>>,
    maximal_cliques: Vec<Clique>,
    node_cliques: Vec<Vec<(usize, usize)>>,
}

impl<T> Graph<T> {
    /// Generates all sub-cliques of a given size from maximal cliques
    pub fn cliques_of_order(&self, order: usize) -> Vec<Clique> {
        let mut result = Vec::new();
        let mut seen = std::collections::HashSet::<Vec<usize>>::new();
        for mc in &self.maximal_cliques {
            if mc.len() >= order {
                for sub in mc.subsets(order) {
                    if seen.insert(sub.members().to_vec()) {
                        result.push(sub);
                    }
                }
            }
        }
        result
    }

    pub fn cliques_containing(&self, node: usize, order: Option<usize>) -> Vec<&Clique> {
        self.node_cliques[node].iter()
            .filter(|(_, o)| order.map_or(true, |want| *o == want))
            .map(|(ci, _)| &self.maximal_cliques[*ci])
            .collect()
    }
    pub fn for_cliques_containing(
        &self, 
        node: usize, 
        order: Option<usize>, 
        mut f: impl FnMut(&Clique),
    ) {
        for &(ci, o) in &self.node_cliques[node] {
            if order.map_or(true, |want| o == want) {
                f(&self.maximal_cliques[ci]);
            }
        }
    }
    pub fn get_node(&self, index: usize) -> &Node<T> {
        &self.nodes[index]
    }
    pub fn get_node_mut(&mut self, index: usize) -> &mut Node<T> {
        &mut self.nodes[index]
    }
}

impl<T: Default> Graph<T> {
    pub fn new(num_nodes: usize) -> Self {
        let mut nodes = Vec::with_capacity(num_nodes);
        for n in 0..num_nodes {
            nodes.push(Node::new(n, T::default(), T::default(), Vec::new()));
        }
        Self {
            nodes,
            maximal_cliques: Vec::new(),
            node_cliques: vec![Vec::new(); num_nodes],
        }
    }

    pub fn add_edge(&mut self, a: usize, b: usize) {
        self.nodes[a].add_edge(b);
        self.nodes[b].add_edge(a);
    }

    pub fn detect_cliques(&mut self) {
        self.maximal_cliques.clear();
        let all: Vec<usize> = (0..self.nodes.len()).collect();
        let mut results = Vec::new();
        Self::bron_kerbosch(&self.nodes, Vec::new(), all, Vec::new(), &mut results);
        self.maximal_cliques = results;
    
        // Precompute per-node lookup
        self.node_cliques = vec![Vec::new(); self.nodes.len()];
        for (ci, clique) in self.maximal_cliques.iter().enumerate() {
            let order = clique.len();
            for &node in clique.members() {
                self.node_cliques[node].push((ci, order));
            }
        }
    }

    fn bron_kerbosch(
        nodes: &[Node<T>],
        r: Vec<usize>,
        mut p: Vec<usize>,
        mut x: Vec<usize>,
        results: &mut Vec<Clique>,
    ) {
        if p.is_empty() && x.is_empty() {
            if r.len() >= 2 {
                results.push(Clique::new(r));
            }
            return;
        }

        let pivot = p.iter().chain(x.iter())
            .max_by_key(|&&v| {
                p.iter().filter(|&&u| nodes[v].neighbors().contains(&u)).count()
            })
            .copied()
            .unwrap();

        let candidates: Vec<usize> = p.iter()
            .filter(|&&v| !nodes[pivot].neighbors().contains(&v))
            .copied()
            .collect();

        for v in candidates {
            let neighbors = nodes[v].neighbors();
            let new_r = [r.clone(), vec![v]].concat();
            let new_p = p.iter().filter(|&&u| neighbors.contains(&u)).copied().collect();
            let new_x = x.iter().filter(|&&u| neighbors.contains(&u)).copied().collect();

            Self::bron_kerbosch(nodes, new_r, new_p, new_x, results);

            p.retain(|&u| u != v);
            x.push(v);
        }
    }

    /// Returns all maximal cliques
    pub fn maximal_cliques(&self) -> &[Clique] {
        &self.maximal_cliques
    }

}

impl<T> Neighborhood for Graph<T> {
    fn neighbors(&self, node: usize) -> &[usize] {
        self.nodes[node].neighbors()
    }

    fn num_nodes(&self) -> usize {
        self.nodes.len()
    }
}