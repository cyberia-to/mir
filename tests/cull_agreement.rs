//! The CPU cull and the GPU cull must decide the same thing.
//!
//! There are now two implementations of "is this particle visible, and how big
//! is it on screen" — one in Rust for small graphs, one as a compute kernel for
//! large ones. Two implementations of one rule is a real cost, and the only
//! thing that makes it affordable is a test that fails when they part company.
//!
//! Tiers matter visually: the tier decides whether a particle is ray-cast or
//! drawn as a disc, and the thresholds are where the two paths meet. A rule
//! that drifts on one side shows up as particles changing character depending
//! on how big the graph happens to be.

use mir::bevy::resources::GraphCamera;
use mir::frame::cull::{CullPass, TierLevel, cull_cpu};

/// Particles spread across depth and angle: some behind the camera, some far
/// out of frame, some straddling the tier thresholds.
fn scene(n: usize) -> (Vec<f32>, Vec<f32>) {
    let mut positions = Vec::with_capacity(n * 3);
    let mut radii = Vec::with_capacity(n);
    for i in 0..n {
        let a = i as f32 * 0.61;
        let d = (i % 17) as f32 / 16.0;
        positions.extend_from_slice(&[
            a.cos() * (400.0 + d * 5000.0),
            a.sin() * (400.0 + d * 5000.0),
            // Straddles the near plane, so some of these are behind the eye.
            (i as f32 - n as f32 * 0.5) * 90.0,
        ]);
        // Radii chosen to land on both sides of every tier threshold.
        radii.push(4.0 + (i % 23) as f32 * 38.0);
    }
    (positions, radii)
}

fn camera(yaw: f32, pitch: f32) -> GraphCamera {
    let mut cam = GraphCamera::default();
    cam.viewport = [1024.0, 640.0];
    cam.yaw = yaw;
    cam.pitch = pitch;
    let f = cam.forward();
    cam.position = [-f[0] * 3000.0, -f[1] * 3000.0, -f[2] * 3000.0];
    cam
}

fn sorted(mut v: Vec<(u32, TierLevel)>) -> Vec<(u32, TierLevel)> {
    v.sort_unstable_by_key(|&(i, _)| i);
    v
}

#[test]
fn cpu_and_gpu_cull_agree() {
    const N: usize = 400;
    let (positions, radii) = scene(N);

    let gpu = mir::gpu::Gpu::open().expect("no GPU for the cull test");
    let pos_buf = gpu.buffer_with_data(bytemuck_f32(&positions)).expect("positions");
    let rad_buf = gpu.buffer_with_data(bytemuck_f32(&radii)).expect("radii");
    let dummy = gpu.buffer(4).expect("dummy bvh");
    let pass = CullPass::new().expect("cull pipeline");

    // Several poses: a cull that agrees only head-on agrees by accident.
    for (yaw, pitch) in [(0.0f32, 0.0f32), (0.8, 0.35), (-2.1, -0.6), (3.0, 1.1)] {
        let cam = camera(yaw, pitch);
        let gpu_cam = cam.to_gpu_camera();

        let on_cpu = sorted(cull_cpu(&positions, &radii, &gpu_cam, N as u32).entries);
        let on_gpu = sorted(
            pass.run(&pos_buf, &rad_buf, &dummy, &gpu_cam, N as u32)
                .expect("gpu cull")
                .entries,
        );

        assert_eq!(
            on_cpu.len(),
            on_gpu.len(),
            "at yaw {yaw} pitch {pitch}: CPU keeps {} particles, GPU keeps {}",
            on_cpu.len(),
            on_gpu.len(),
        );
        for (a, b) in on_cpu.iter().zip(on_gpu.iter()) {
            assert_eq!(
                a, b,
                "at yaw {yaw} pitch {pitch}: particle {} is tier {:?} on the CPU \
                 and {:?} on the GPU",
                a.0, a.1, b.1,
            );
        }
    }
}

fn bytemuck_f32(v: &[f32]) -> &[u8] {
    // Same reinterpretation the frame path uses to upload these buffers.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}
