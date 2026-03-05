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
    let width = 64;
    let height = 64;
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

    let pairwise = TablePotential::pairwise(&[
        //   W     F     D     R
        vec![0.6, 0.5, 0.05, 0.05],
        vec![0.5, 0.6, 0.4,  0.05],
        vec![0.05, 0.4, 0.6,  0.5],
        vec![0.05, 0.05, 0.5,  0.6],
    ]).unwrap();

    let mut mrf = MRF::<usize>::builder()
        .graph(graph)
        .potential(pairwise)
        .build()
        .unwrap();

    let annealer = LinearAnnealer::new(5.0, 0.1, 0.25);
    let sampler = GibbsSampler::new(sweeps, annealer);
    let proposal = DiscreteProposal::new(num_labels);

    std::fs::create_dir_all("frames").unwrap();

    save_frame(&mrf, width, height, "frames/frame_000.png");

    let opts = RunOptions {
        seed: Some(42),
    };

    sampler.run(&mut mrf, &proposal, opts).unwrap();

    println!("Done! Frames saved to frames/");
}