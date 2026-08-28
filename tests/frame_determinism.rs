//! The same scene, twice, must be the same pixels.
//!
//! Nothing in the renderer looks random, and that is the problem: the sources
//! of frame-to-frame variation here were all incidental. The cull kernel
//! appends visible particles through an atomic counter, so their order is
//! whatever the GPU scheduled. Edges were gathered by iterating a `HashSet`
//! built fresh every frame, and a Rust hash set's iteration order depends on
//! the key its instance was seeded with. A depth sort broke ties arbitrarily.
//!
//! None of that would matter if drawing were order-independent. It is not:
//! splats and edge glow both composite with alpha, so reordering them
//! reorders the blend and changes the image. The result was a graph that
//! shimmered wherever links crossed — and only while the camera moved, since
//! a still camera short-circuits the whole path and recomputes nothing. That
//! is a nasty shape for a bug: it disappears exactly when you stop to look.

use mir::bevy::resources::GraphCamera;
use mir::frame::cull::TierLevel;
use mir::frame::paint::{PaintPass, edge_segments, sort_by_depth};

const W: u32 = 320;
const H: u32 = 240;

fn scene() -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<(u32, TierLevel)>) {
    let mut positions = Vec::new();
    let mut radii = Vec::new();
    let mut colors = Vec::new();
    let mut visible = Vec::new();
    // A spread of particles at assorted depths, several sharing one exactly —
    // equal depths are where a sort's tie-breaking starts to show.
    for i in 0..12u32 {
        // 3 sits exactly on 2 and 7 exactly on 6, so two pairs share a depth
        // to the last bit. Ties are where a sort's tie-breaking stops being
        // theoretical.
        let a = match i { 3 => 2u32, 7 => 6, other => other } as f32 * 0.7;
        positions.extend_from_slice(&[a.cos() * 700.0, a.sin() * 500.0,
                                      (match i { 3 => 2u32, 7 => 6, o => o } % 4) as f32 * 300.0]);
        radii.push(180.0);
        colors.extend_from_slice(&[0.2, 0.5, 1.0]);
        visible.push((i, if i % 2 == 0 { TierLevel::T2 } else { TierLevel::T3 }));
    }
    (positions, radii, colors, visible)
}

fn camera() -> GraphCamera {
    let mut cam = GraphCamera::default();
    cam.viewport = [W as f32, H as f32];
    cam.yaw = 0.6;
    cam.pitch = 0.25;
    let f = cam.forward();
    cam.position = [-f[0] * 3000.0, -f[1] * 3000.0, -f[2] * 3000.0];
    cam
}

fn render(visible: &[(u32, TierLevel)]) -> Vec<u8> {
    let (positions, radii, colors, _) = scene();
    let cam = camera();
    let gpu_cam = cam.to_gpu_camera();

    // Deliberately *not* normalised here: the point is that the renderer does
    // not need the caller to hand it a tidy order. Edges are gathered in the
    // order the particles arrive, exactly as the frame path does it, so any
    // order-dependence downstream shows up in the pixels.
    let ordered: Vec<(u32, TierLevel)> = visible.to_vec();
    let sorted = sort_by_depth(&ordered, &positions, &gpu_cam);

    let mut zipped: Vec<(u32, u32, f32)> = ordered
        .iter()
        .filter(|&&(i, _)| i + 1 < 12)
        .map(|&(i, _)| (i, i + 1, 0.8f32))
        .collect();
    zipped.sort_unstable_by_key(|&(p, q, _)| (p, q));
    let edges: Vec<(u32, u32)> = zipped.iter().map(|&(p, q, _)| (p, q)).collect();
    let weights: Vec<f32> = zipped.iter().map(|&(_, _, w)| w).collect();
    let segments = edge_segments(&edges, &weights, &positions, &gpu_cam, [W, H]);

    let gpu = mir::gpu::Gpu::open().expect("no GPU for the render test");
    let queue = gpu.new_command_queue().expect("command queue");
    let paint = PaintPass::new().expect("paint pipeline");
    let dst = gpu.buffer((W as usize) * (H as usize) * 4).expect("frame buffer");
    let cmd = queue.commands().expect("commands");
    paint
        .draw(&sorted, &ordered, &positions, &radii, &colors,
              &segments, &gpu_cam, [W, H], &dst, &cmd)
        .expect("draw");
    cmd.submit();

    let mut pixels = vec![0u8; (W as usize) * (H as usize) * 4];
    mir::gpu::FrameReader::new().fetch(&gpu, &queue, &dst, &mut pixels);
    pixels
}

fn differing_pixels(a: &[u8], b: &[u8]) -> usize {
    a.chunks_exact(4).zip(b.chunks_exact(4)).filter(|(p, q)| p != q).count()
}

#[test]
fn the_same_scene_twice_is_the_same_frame() {
    let (_, _, _, visible) = scene();
    let a = render(&visible);
    let b = render(&visible);
    let d = differing_pixels(&a, &b);
    assert_eq!(d, 0, "two renders of one scene differ in {d} pixels");
}

/// The cull hands its particles over in whatever order the GPU produced them.
/// That order must not reach the image.
#[test]
fn the_order_the_cull_returns_particles_in_does_not_matter() {
    let (_, _, _, visible) = scene();
    let baseline = render(&visible);

    // Three orders no cull would produce on purpose, and one it might.
    let mut reversed = visible.clone();
    reversed.reverse();
    let mut rotated = visible.clone();
    rotated.rotate_left(5);
    let mut interleaved: Vec<_> = visible.iter().step_by(2).copied().collect();
    interleaved.extend(visible.iter().skip(1).step_by(2).copied());

    for (name, order) in [
        ("reversed", reversed),
        ("rotated", rotated),
        ("interleaved", interleaved),
    ] {
        let d = differing_pixels(&baseline, &render(&order));
        assert_eq!(
            d, 0,
            "the {name} particle order draws {d} pixels differently. Something \
             downstream is order-dependent — which means the image depends on \
             GPU scheduling, and the graph shimmers whenever the camera moves."
        );
    }
}

/// Equal depths must not swap between calls: swapping two alpha-composited
/// particles changes the pixels under them.
#[test]
fn the_depth_sort_breaks_ties_the_same_way_every_time() {
    let (positions, _, _, visible) = scene();
    let cam = camera().to_gpu_camera();

    let baseline = sort_by_depth(&visible, &positions, &cam);
    let mut reversed = visible.clone();
    reversed.reverse();

    assert_eq!(
        baseline,
        sort_by_depth(&reversed, &positions, &cam),
        "the depth sort gives a different answer for the same particles in a \
         different order — so the order the cull happened to return them in \
         reaches the pixels"
    );
}
