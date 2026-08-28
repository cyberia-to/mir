//! A particle must be drawn as a round ball, where the projection says it is,
//! at the same size whichever tier draws it — and all of that must hold with
//! the camera pointing somewhere other than straight down an axis.
//!
//! That last clause is the whole reason this file exists in its current shape.
//! An earlier version of it tested only the default camera, which looks along
//! -Z with no yaw or pitch. With that camera `view_proj[0][0]` happens to equal
//! the focal scale exactly, so a shader reading the focal length out of the
//! matrix passed every check — and drew ellipses the moment anyone turned the
//! graph, because those entries are the focal scale times `right.x` and `up.y`.
//! A renderer test that only ever looks down an axis is testing the one pose
//! where the interesting bugs are invisible.

use mir::bevy::resources::GraphCamera;
use mir::frame::cull::TierLevel;
use mir::frame::paint::PaintPass;

const W: u32 = 480;
const H: u32 = 200;
const DIST: f32 = 3000.0;

/// Blue at least this bright is inside the particle's silhouette. The dimmest
/// the lit surface gets is its ambient term (0.2 of full blue ≈ 51); outside
/// the silhouette there is nothing but black.
const INSIDE: u8 = 25;

struct Rendered {
    pixels: Vec<u8>,
}

impl Rendered {
    /// Bounding box of the silhouette, as (min_x, min_y, max_x, max_y).
    fn silhouette(&self) -> Option<(u32, u32, u32, u32)> {
        let (mut x0, mut y0, mut x1, mut y1) = (u32::MAX, u32::MAX, 0u32, 0u32);
        let mut any = false;
        for y in 0..H {
            for x in 0..W {
                if self.pixels[((y * W + x) * 4 + 2) as usize] >= INSIDE {
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

    fn extent(&self) -> (f32, f32, f32, f32) {
        let (x0, y0, x1, y1) = self.silhouette().expect("particle was not drawn at all");
        (
            (x1 - x0 + 1) as f32,
            (y1 - y0 + 1) as f32,
            (x0 + x1) as f32 * 0.5,
            (y0 + y1) as f32 * 0.5,
        )
    }
}

/// A camera at `DIST` from the origin, looking at it from the given angles.
fn camera_at(yaw: f32, pitch: f32) -> GraphCamera {
    let mut cam = GraphCamera::default();
    cam.viewport = [W as f32, H as f32];
    cam.yaw = yaw;
    cam.pitch = pitch;
    let f = cam.forward();
    cam.position = [-f[0] * DIST, -f[1] * DIST, -f[2] * DIST];
    cam
}

/// One blue particle at `pos`, drawn in `tier`, on an otherwise black frame.
fn render_one(cam: &GraphCamera, pos: [f32; 3], r: f32, tier: TierLevel) -> Rendered {
    let gpu = mir::gpu::Gpu::open().expect("no GPU for the render test");
    let queue = gpu.new_command_queue().expect("command queue");
    let paint = PaintPass::new().expect("paint pipeline");

    let camera = cam.to_gpu_camera();
    let dst = gpu
        .buffer((W as usize) * (H as usize) * 4)
        .expect("frame buffer");
    let cmd = queue.commands().expect("commands");
    paint
        .draw(&[0u32], &[(0u32, tier)], &pos.to_vec(), &[r], &[0.0, 0.0, 1.0],
              &[], &camera, [W, H], &dst, &cmd)
        .expect("draw");
    cmd.submit();

    let mut pixels = vec![0u8; (W as usize) * (H as usize) * 4];
    mir::gpu::FrameReader::new().fetch(&gpu, &queue, &dst, &mut pixels);
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

fn assert_round(what: &str, w: f32, h: f32, tol: std::ops::RangeInclusive<f32>) {
    assert!(
        w > 8.0 && h > 8.0,
        "{what}: silhouette is {w}x{h} px — too small to judge shape"
    );
    let ratio = w / h;
    assert!(
        tol.contains(&ratio),
        "{what}: drawn {w}x{h} px, aspect {ratio:.2}, not round. The viewport \
         is {W}x{H} (aspect {:.2}); stretching toward that number is the \
         signature of a focal scale read out of view_proj instead of the camera.",
        W as f32 / H as f32,
    );
}

#[test]
fn a_solid_particle_is_round() {
    let cam = camera_at(0.0, 0.0);
    let (w, h, ..) = render_one(&cam, [0.0, 0.0, 0.0], 500.0, TierLevel::T2).extent();
    assert_round("head-on", w, h, 0.9..=1.1);
}

/// The regression test for spheres drawn as ellipses. Nothing about the shader
/// changes between this and the test above except where the camera is looking.
#[test]
fn a_solid_particle_is_round_with_the_camera_turned() {
    for (yaw, pitch) in [(0.7f32, 0.4f32), (-1.2, -0.5), (2.4, 0.9)] {
        let cam = camera_at(yaw, pitch);
        let (w, h, ..) = render_one(&cam, [0.0, 0.0, 0.0], 500.0, TierLevel::T2).extent();
        assert_round(&format!("yaw {yaw} pitch {pitch}"), w, h, 0.9..=1.1);
    }
}

#[test]
fn a_solid_particle_lands_where_it_is_projected() {
    // Off-centre and off-axis: at the centre of the frame a misplaced ray
    // still hits, so the bug hides there.
    let cam = camera_at(0.5, 0.3);
    let pos = [600.0, -250.0, 200.0];
    let (_, _, cx, cy) = render_one(&cam, pos, 500.0, TierLevel::T2).extent();
    let (px, py) = projected(&cam, pos);

    let (dx, dy) = ((cx - px).abs(), (cy - py).abs());
    assert!(
        dx <= 2.0 && dy <= 2.0,
        "particle projects to ({px:.1}, {py:.1}) but was drawn at \
         ({cx:.1}, {cy:.1}) — off by ({dx:.1}, {dy:.1}) px. The ray-cast and \
         the projection disagree about where this particle is."
    );
}

/// Tier decides *how* a particle is drawn, never how big it looks. When the two
/// paths disagreed on size, zooming across the threshold made every node jump
/// between two diameters — which reads as the graph flickering.
#[test]
fn the_two_draw_paths_agree_on_size() {
    let cam = camera_at(0.4, 0.2);
    let solid = render_one(&cam, [0.0, 0.0, 0.0], 500.0, TierLevel::T2).extent();
    let splat = render_one(&cam, [0.0, 0.0, 0.0], 500.0, TierLevel::T3).extent();

    let (dw, dh) = ((solid.0 - splat.0).abs(), (solid.1 - splat.1).abs());
    assert!(
        dw <= 2.0 && dh <= 2.0,
        "solid draws the particle {}x{} px and splat draws it {}x{} px \
         — a jump of ({dw}, {dh}) px when a particle crosses the tier \
         threshold, which is visible as flicker while zooming.",
        solid.0, solid.1, splat.0, splat.1,
    );
    assert!(
        (solid.2 - splat.2).abs() <= 1.5 && (solid.3 - splat.3).abs() <= 1.5,
        "the two paths also disagree about where the particle is: solid at \
         ({:.1}, {:.1}), splat at ({:.1}, {:.1})",
        solid.2, solid.3, splat.2, splat.3,
    );
}
