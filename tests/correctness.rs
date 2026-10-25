//! Statistical correctness tests for the Gibbs sampler.
//!
//! The idea: on a graph small enough to enumerate every configuration, we can
//! compute the exact Boltzmann distribution p(x) ∝ exp(-E(x)/T) from
//! `MRF::energy`, then check that the sampler's long-run empirical histogram
//! converges to it. If node_energy, clique lookup, or the conditional sampling
//! step has a bug, these distributions will disagree.

use mrf::*;

const L: usize = 3; // labels
const N: usize = 4; // 2x2 grid -> 3^4 = 81 states

fn small_grid() -> Graph<usize> {
    let mut g: Graph<usize> = Graph::new(N);
    // 0 1
    // 2 3
    g.add_edge(0, 1);
    g.add_edge(2, 3);
    g.add_edge(0, 2);
    g.add_edge(1, 3);
    g.detect_cliques();
    g
}

fn decode(mut idx: usize) -> [usize; N] {
    let mut s = [0; N];
    for v in s.iter_mut() {
        *v = idx % L;
        idx /= L;
    }
    s
}

fn encode(mrf: &MRF<usize>) -> usize {
    (0..N).rev().fold(0, |acc, i| acc * L + *mrf.graph().get_node(i).state())
}

fn build(pairwise: TablePotential, unary: TablePotential) -> MRF<usize> {
    MRF::<usize>::builder()
        .graph(small_grid())
        .potential(pairwise)
        .potential(unary)
        .build()
        .unwrap()
}

fn potentials() -> (TablePotential, TablePotential) {
    // Asymmetric enough that a wrong conditional would be visible.
    let pw = TablePotential::new(vec![0.0, 1.0, 2.0, 1.0, 0.0, 0.5, 2.0, 0.5, 0.0], L, 2).unwrap();
    let un = TablePotential::unary(vec![0.0, 0.3, -0.4]);
    (pw, un)
}

fn exact_distribution(t: f64) -> Vec<f64> {
    let (pw, un) = potentials();
    let mut mrf = build(pw, un);
    let mut p: Vec<f64> = (0..L.pow(N as u32))
        .map(|idx| {
            for (i, &s) in decode(idx).iter().enumerate() {
                mrf.graph_mut().get_node_mut(i).set_state(s);
            }
            (-mrf.energy() / t).exp()
        })
        .collect();
    let z: f64 = p.iter().sum();
    p.iter_mut().for_each(|x| *x /= z);
    p
}

fn empirical_distribution(t: f64, sweeps: usize, burn_in: usize) -> Vec<f64> {
    let (pw, un) = potentials();
    let mut mrf = build(pw, un);
    let mut counts = vec![0usize; L.pow(N as u32)];
    GibbsSampler::new(sweeps, ConstantAnnealer::new(t))
        .run_with(&mut mrf, &DiscreteProposal::new(L), RunOptions { seed: Some(7) }, |i, m| {
            if i > burn_in {
                counts[encode(m)] += 1;
            }
        })
        .unwrap();
    let n = (sweeps - burn_in) as f64;
    counts.into_iter().map(|c| c as f64 / n).collect()
}

fn total_variation(p: &[f64], q: &[f64]) -> f64 {
    0.5 * p.iter().zip(q).map(|(a, b)| (a - b).abs()).sum::<f64>()
}

#[test]
fn gibbs_matches_exact_boltzmann_t1() {
    let exact = exact_distribution(1.0);
    let emp = empirical_distribution(1.0, 200_000, 1_000);
    let tv = total_variation(&exact, &emp);
    assert!(tv < 0.02, "TV distance {tv:.4} too large at T=1");
}

#[test]
fn gibbs_matches_exact_boltzmann_t_half() {
    let exact = exact_distribution(0.5);
    let emp = empirical_distribution(0.5, 200_000, 1_000);
    let tv = total_variation(&exact, &emp);
    assert!(tv < 0.02, "TV distance {tv:.4} too large at T=0.5");
}

/// Semantic check: `TablePotential::pairwise` is documented as taking
/// *weights* (bigger = more compatible). A strongly agreeing matrix must
/// therefore give a uniform field LOWER energy than a checkerboard.
#[test]
fn pairwise_weights_favor_agreement() {
    let pw = TablePotential::pairwise(&[vec![1.0, 0.01], vec![0.01, 1.0]]).unwrap();
    let mut mrf = MRF::<usize>::builder().graph(small_grid()).potential(pw).build().unwrap();

    for i in 0..N { mrf.graph_mut().get_node_mut(i).set_state(0); }
    let uniform = mrf.energy();
    for (i, s) in [0, 1, 1, 0].into_iter().enumerate() { mrf.graph_mut().get_node_mut(i).set_state(s); }
    let checker = mrf.energy();

    assert!(uniform < checker, "uniform E={uniform:.2} should be < checkerboard E={checker:.2}");
}

/// Local/global consistency: the change in node_energy when flipping one node
/// must equal the change in total energy.
#[test]
fn node_energy_delta_matches_total_delta() {
    let (pw, un) = potentials();
    let mut mrf = build(pw, un);
    for (i, s) in [0, 2, 1, 1].into_iter().enumerate() { mrf.graph_mut().get_node_mut(i).set_state(s); }
    for node in 0..N {
        let cur = *mrf.graph().get_node(node).state();
        for cand in 0..L {
            let local = mrf.node_energy_with(node, &cand) - mrf.node_energy(node);
            let e0 = mrf.energy();
            mrf.graph_mut().get_node_mut(node).set_state(cand);
            let global = mrf.energy() - e0;
            mrf.graph_mut().get_node_mut(node).set_state(cur);
            assert!((local - global).abs() < 1e-9, "node {node} -> {cand}: local {local} vs global {global}");
        }
    }
}
