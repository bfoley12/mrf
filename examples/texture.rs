use mrf::*;
use image::RgbImage;
use rand::{SeedableRng, RngExt};

fn label_to_color(label: usize) -> [u8; 3] {
    match label {
        0 => [30, 90, 200],    // water — blue
        1 => [34, 139, 34],    // grass — green
        2 => [139, 90, 43],    // dirt — brown
        3 => [128, 128, 128],  // stone — gray
        _ => [0, 0, 0],
    }
}

fn save_frame(mrf: &MRF<usize>, width: usize, height: usize, path: &str) {
    let mut img = RgbImage::new(width as u32, height as u32);
    for i in 0..mrf.num_nodes() {
        let x = (i % width) as u32;
        let y = (i / width) as u32;
        let color = label_to_color(*mrf.graph().get_node(i).state());
        img.put_pixel(x, y, image::Rgb(color));
    }
    img.save(path).unwrap();
}

fn main() {
    let width = 128;
    let height = 128;
    let sweeps = 100;
    let num_labels = 4;

    // Build grid graph
    let mut graph: Graph<usize> = Graph::new(width * height);
    for r in 0..height {
        for c in 0..width {
            let i = r * width + c;
            if c + 1 < width { graph.add_edge(i, i + 1); }
            if r + 1 < height { graph.add_edge(i, i + width); }
        }
    }
    graph.detect_cliques();

    // Randomize initial state
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    for i in 0..graph.num_nodes() {
        graph.get_node_mut(i).set_state(rng.random_range(0..num_labels));
    }

    // Weights: larger = more compatible. Labels form an "elevation" chain
    // W - F - D - R: each label only tolerates its immediate neighbours, so
    // regions nest like contour lines (water, then a grassy coast, then
    // dirt, then stone peaks) instead of forming an arbitrary 2-way split.
    let pairwise = TablePotential::pairwise(&[
        //    W     F     D     R
        vec![1.0,  0.25, 0.01, 0.01],
        vec![0.25, 1.0,  0.25, 0.01],
        vec![0.01, 0.25, 1.0,  0.25],
        vec![0.01, 0.01, 0.25, 1.0 ],
    ]).unwrap();

    // Unary energies (lower = more likely). Middle labels have two friendly
    // neighbours in the chain, so they win on entropy alone and the ends die
    // out. These biases push back: lots of ocean, a few stone peaks.
    let unary = TablePotential::unary(vec![-0.25, 0.0, 0.0, -0.4]);

    let mut mrf = MRF::<usize>::builder()
        .graph(graph)
        .potential(pairwise)
        .potential(unary)
        .build()
        .unwrap();

    // Cool from disordered to just below the ordering transition. Below it the
    // model coarsens: blobs merge until one phase wins (that's the correct
    // equilibrium for a local pairwise MRF). "Continents" are a snapshot of
    // that coarsening, so `sweeps` controls feature size more than quality.
    let annealer = ExponentialAnnealer::new(3.0, 0.98, 1.2);
    let sampler = GibbsSampler::new(sweeps, annealer);
    let proposal = DiscreteProposal::new(num_labels);

    std::fs::create_dir_all("frames").unwrap();

    let opts = RunOptions { seed: Some(42) };

    sampler.run_with(&mut mrf, &proposal, opts, |sweep, m| {
        if sweep % 10 == 0 || sweep == sweeps {
            save_frame(m, width, height, &format!("frames/frame_{sweep:03}.png"));
        }
    }).unwrap();

    println!("Done! Frames saved to frames/ (final energy {:.1})", mrf.energy());
}
