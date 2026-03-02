use crate::graph::Neighborhood;
use crate::graph::node::Node;
use crate::graph::clique::Clique;

pub struct Graph<T> {
    nodes: Vec<Node<T>>,
    maximal_cliques: Vec<Clique>,
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
        }
    }

    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
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

    pub fn cliques_containing(&self, node: usize, order: Option<usize>) -> Vec<Clique> {
        match order {
            Some(o) => self.cliques_of_order(o)
                .into_iter()
                .filter(|c| c.contains(node))
                .collect(),
            None => self.maximal_cliques.iter()
                .filter(|c| c.contains(node))
                .cloned()
                .collect(),
        }
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