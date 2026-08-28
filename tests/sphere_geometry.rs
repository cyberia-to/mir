//! A particle must be drawn as a round ball, where the projection says it is.
//!
//! The paint pass draws solid particles by ray-casting, which means it needs
//! the camera's position and basis — quantities that are *not* recoverable
//! from the view-projection matrix, however much its columns look like a
//! basis. When that was got wrong the spheres still appeared, just squashed to
//! the viewport's aspect and nowhere near the particles they belonged to; the
//! graph looked like it had a second, ghostly set of nodes.
//!
//! Nothing about that is visible to a compiler, and on a still screenshot it
//! reads as a style choice. So it gets measured: render one particle, find the
//! pixels it covers, and check where they are and what shape they make.
//!
//! The viewport is deliberately not square — an aspect bug is invisible at 1:1.

use mir::bevy::resources::GraphCamera;
use mir::frame::cull::TierLevel;
use mir::frame::paint::PaintPass;

const W: u32 = 480;
const H: u32 = 200;

/// Pixels this bright in blue are inside the sphere's silhouette. The dimmest
/// the lit surface gets is its ambient term (0.2 of full blue = 51); the
/// brightest the halo behind it reaches just outside the silhouette is about
/// 5. Anywhere between the two separates them cleanly.
const INSIDE: u8 = 25;

struct Rendered {
    pixels: Vec<u8>,
}

impl Rendered {
    fn blue(&self, x: u32, y: u32) -> u8 {
        self.pixels[((y * W + x) * 4 + 2) as usize]
    }

    /// Bounding box of the silhouette, as (min_x, min_y, max_x, max_y).
    fn silhouette(&self) -> Option<(u32, u32, u32, u32)> {
        let (mut x0, mut y0, mut x1, mut y1) = (u32::MAX, u32::MAX, 0u32, 0u32);
        let mut any = false;
        for y in 0..H {
            for x in 0..W {
                if self.blue(x, y) >= INSIDE {
                    any = true;
                    x0 = x0.min(x);
                    y0 = y0.min(y);
                    x1 = x1.max(x);
                    y1 = y1.max(y);
                }
            }
        }
        any.then_some((x0, y0, x1, y1))
    }
}

/// One blue particle of radius `r` at `pos`, drawn solid, on a black frame.
fn render_one(pos: [f32; 3], r: f32) -> Rendered {
    let gpu = mir::gpu::Gpu::open().expect("no GPU for the render test");
    let queue = gpu.new_command_queue().expect("command queue");
    let paint = PaintPass::new().expect("paint pipeline");

    let mut cam = GraphCamera::default();
    cam.viewport = [W as f32, H as f32];
    let camera = cam.to_gpu_camera();

    let positions = pos.to_vec();
    let radii = vec![r];
    let colors = vec![0.0, 0.0, 1.0];
    let visible = vec![(0u32, TierLevel::T2)];
    let sorted = vec![0u32];

    let dst = gpu
        .buffer((W as usize) * (H as usize) * 4)
        .expect("frame buffer");
    let cmd = queue.commands().expect("commands");
    paint
        .draw(&sorted, &visible, &positions, &radii, &colors,
              &[], &camera, [W, H], &dst, &cmd)
        .expect("draw");
    cmd.submit();

    let mut pixels = vec![0u8; (W as usize) * (H as usize) * 4];
    let mut reader = mir::gpu::FrameReader::new();
    reader.fetch(&gpu, &queue, &dst, &mut pixels);
    Rendered { pixels }
}

/// Where the camera's own matrix says the particle lands, in pixels.
fn projected(cam: &GraphCamera, pos: [f32; 3]) -> (f32, f32) {
    let m = cam.view_proj();
    let [x, y, z] = pos;
    let w = m[0][3] * x + m[1][3] * y + m[2][3] * z + m[3][3];
    let cx = (m[0][0] * x + m[1][0] * y + m[2][0] * z + m[3][0]) / w;
    let cy = (m[0][1] * x + m[1][1] * y + m[2][1] * z + m[3][1]) / w;
    (
        (cx * 0.5 + 0.5) * W as f32,
        (1.0 - (cy * 0.5 + 0.5)) * H as f32,
    )
}

#[test]
fn a_solid_particle_is_round() {
    let f = render_one([0.0, 0.0, 0.0], 500.0);
    let (x0, y0, x1, y1) = f.silhouette().expect("the particle was not drawn at all");
    let (w, h) = ((x1 - x0 + 1) as f32, (y1 - y0 + 1) as f32);

    assert!(
        w > 8.0 && h > 8.0,
        "silhouette is {w}x{h} px — too small to judge; the test's radius or \
         camera distance drifted"
    );
    // A ball is as wide as it is tall. Squashing to the viewport's aspect is
    // the signature of a ray built from the projection instead of the camera.
    let ratio = w / h;
    assert!(
        (0.9..=1.1).contains(&ratio),
        "particle drawn {w}x{h} px — aspect {ratio:.2}, not round. \
         The viewport is {W}x{H} (aspect {:.2}), which is what a ray \
         reconstructed from view_proj's columns would stamp onto it.",
        W as f32 / H as f32,
    );
}

#[test]
fn a_solid_particle_lands_where_it_is_projected() {
    // Off-centre: at the centre of the frame a misplaced ray still hits, so
    // the bug hides there. Away from it the error grows with the angle.
    let pos = [600.0, -250.0, 0.0];
    let f = render_one(pos, 500.0);
    let (x0, y0, x1, y1) = f.silhouette().expect("the particle was not drawn at all");
    let (cx, cy) = (
        (x0 + x1) as f32 * 0.5,
        (y0 + y1) as f32 * 0.5,
    );

    let mut cam = GraphCamera::default();
    cam.viewport = [W as f32, H as f32];
    let (px, py) = projected(&cam, pos);

    let (dx, dy) = ((cx - px).abs(), (cy - py).abs());
    assert!(
        dx <= 2.0 && dy <= 2.0,
        "particle projects to ({px:.1}, {py:.1}) but was drawn at \
         ({cx:.1}, {cy:.1}) — off by ({dx:.1}, {dy:.1}) px. The ray-cast and \
         the projection disagree about where this particle is."
    );
}

/// The same particle, drawn twice at two viewport shapes, must keep its size
/// relative to the frame. This is the "a Mac and a phone draw the same graph"
/// property, reduced to something a test can hold.
#[test]
fn a_solid_particle_keeps_its_shape_off_axis() {
    // Near a corner, where any error in the ray basis is at its largest.
    let f = render_one([900.0, 380.0, 0.0], 500.0);
    let (x0, y0, x1, y1) = f.silhouette().expect("the particle was not drawn at all");
    let (w, h) = ((x1 - x0 + 1) as f32, (y1 - y0 + 1) as f32);
    let ratio = w / h;
    assert!(
        (0.85..=1.18).contains(&ratio),
        "off-axis particle drawn {w}x{h} px — aspect {ratio:.2}. Some \
         foreshortening is real at the edge of a perspective frame; this is \
         far past it."
    );
}
