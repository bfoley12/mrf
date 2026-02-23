use mrf::*;
use rand::SeedableRng;
use image::RgbImage;

fn label_to_color(label: &Label) -> [u8; 3] {
    match label.as_index() {
        0 => [34, 139, 34],    // grass — green
        1 => [139, 90, 43],    // dirt — brown
        2 => [128, 128, 128],  // stone — gray
        3 => [30, 90, 200],    // water — blue
        _ => [0, 0, 0],
    }
}

fn save_frame(field: &[Label], width: usize, height: usize, path: &str) {
    let mut img = RgbImage::new(width as u32, height as u32);
    for (i, label) in field.iter().enumerate() {
        let x = (i % width) as u32;
        let y = (i / width) as u32;
        let color = label_to_color(label);
        img.put_pixel(x, y, image::Rgb(color));
    }
    img.save(path).unwrap();
}

fn main() {
    let width = 64;
    let height = 64;
    let sweeps = 100;

    let labels = DiscreteLabels::new(4);
    let grid = Grid2D::new(width, height, Connectivity::Eight);
    let pairwise = MatrixPairwise::new(&vec![
        vec![1.0, 0.3, 0.3, 0.0],
        vec![0.3, 1.0, 0.2, 0.3],
        vec![0.3, 0.2, 1.0, 0.3],
        vec![0.0, 0.3, 0.3, 1.0],
    ]).unwrap();

    let mrf = MRF::builder()
        .state_space(labels)
        .neighborhood(grid)
        .pairwise(pairwise)
        .build();

    let sampler = GibbsSampler::new(sweeps, 2.0);
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);

    // Initialize randomly
    let mut field = mrf.random_init(&mut rng);

    // Create output directory
    std::fs::create_dir_all("frames").unwrap();

    // Save each frame
    save_frame(&field, width, height, "frames/frame_000.png");

    for i in 0..sweeps {
        sampler.sweep(
            mrf.state_space(),
            mrf.neighborhood(),
            mrf.unary(),
            mrf.pairwise(),
            &mut field,
            &mut rng,
        );
        save_frame(&field, width, height, &format!("frames/frame_{:03}.png", i + 1));
        println!("Sweep {}/{}", i + 1, sweeps);
    }

    println!("Done! Frames saved to frames/");
}