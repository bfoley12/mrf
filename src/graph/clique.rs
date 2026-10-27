/// A Clique is a group of indices that are considered neighbors
/// field 'members' is a vec of flattened matrix indices of each 
/// member of the Clique
#[derive(Clone)]
pub struct Clique {
    members: Vec<usize>,
}

impl Clique {
    pub fn new(mut members: Vec<usize>) -> Self {
        // Sort improve later access (frequent)
        members.sort_unstable();
        // Dedup to remove any erroneous duplicate members
        // (a node can only join this clique once)
        members.dedup();
        Self { members }
    }

    pub fn len(&self) -> usize {
        self.members.len()
    }
    
    pub fn is_empty(&self) -> bool {
        self.members.len() == 0
    }
    
    pub fn members(&self) -> &[usize] {
        &self.members
    }

    pub fn contains(&self, node: usize) -> bool {
        self.members.binary_search(&node).is_ok()
    }

    pub fn intersect(&self, other: &Clique) -> Clique {
        let (mut i, mut j) = (0, 0);
        let mut result = Vec::new();
        while i < self.len() && j < other.len() {
            match self.members[i].cmp(&other.members[j]) {
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal => {
                    result.push(self.members[i]);
                    i += 1;
                    j += 1;
                }
            }
        }
        Clique::new(result)
    }

    pub fn insert(&mut self, node: usize) {
        match self.members.binary_search(&node) {
            // Nodes cannot join a clique twice
            Ok(_) => {}
            Err(pos) => self.members.insert(pos, node),
        }
    }
    
    /// Find all subsets of Cliques within this Clique, limited to Cliques of size 'size'
    pub fn subsets(&self, size: usize) -> Vec<Clique> {
            let members = self.members();
            let mut result = Vec::new();
            Self::combinations(members, size, 0, &mut Vec::new(), &mut result);
            result
        }
    
    /// Find all combinatinations of size 'size' from 'members'
    fn combinations(
        members: &[usize],
        size: usize,
        start: usize,
        current: &mut Vec<usize>,
        result: &mut Vec<Clique>,
    ) {
        if current.len() == size {
            result.push(Clique::new(current.clone()));
            return;
        }
        for i in start..members.len() {
            current.push(members[i]);
            Self::combinations(members, size, i + 1, current, result);
            current.pop();
        }
    }
}
