//! 6DOF camera control: WASD+QE movement, mouse look, scroll τ zoom.
//! §9.1 — §9.5 navigation.

use bevy::ecs::message::MessageReader;
use bevy::input::mouse::{MouseMotion, MouseWheel};
use bevy::prelude::*;
use super::resources::{GraphCamera, WarpAnim};

const SCROLL_SENS: f32 = 0.1;
const TAU_MIN:     f32 = 0.01;
const TAU_MAX:     f32 = 100.0;
const MOVE_SPEED:  f32 = 500.0;
const LOOK_SENS:   f32 = 0.003;
const DAMPING:     f32 = 8.0;

/// §9.5: τ₀ = 0.1, α = 2, R_scene = 1000.
const TAU0:    f32 = 0.1;
const ALPHA:   f32 = 2.0;
const R_SCENE: f32 = 1000.0;

pub fn update_camera(
    mut cam:    ResMut<GraphCamera>,
    keys:       Res<ButtonInput<KeyCode>>,
    mut motion: MessageReader<MouseMotion>,
    mut scroll: MessageReader<MouseWheel>,
    time:       Res<Time>,
    windows:    Query<&Window>,
) {
    let dt = time.delta_secs();

    if let Ok(win) = windows.single() {
        cam.viewport = [win.width(), win.height()];
    }

    // Advance warp animation (§9.2). Extract all warp data first to avoid borrow conflict.
    if cam.warp.is_some() {
        let (from_pos, to_pos, to_yaw, to_pitch, elapsed, duration) = {
            let w = cam.warp.as_mut().unwrap();
            w.elapsed = (w.elapsed + dt).min(w.duration);
            (w.from_pos, w.to_pos, w.to_yaw, w.to_pitch, w.elapsed, w.duration)
        };
        let t = smooth_step(elapsed / duration);
        for i in 0..3 { cam.position[i] = lerp(from_pos[i], to_pos[i], t); }
        let cur_yaw   = cam.yaw;
        let cur_pitch = cam.pitch;
        cam.yaw   = lerp_angle(cur_yaw,   to_yaw,   t);
        cam.pitch = lerp(cur_pitch, to_pitch, t);
        if elapsed >= duration { cam.warp = None; }
        update_tau_from_position(&mut cam);
        return;
    }

    for ev in scroll.read() {
        let factor = 1.0 - ev.y * SCROLL_SENS;
        cam.tau_target = (cam.tau_target * factor).clamp(TAU_MIN, TAU_MAX);
    }
    let gap = cam.tau_target - cam.tau;
    cam.tau += gap * (DAMPING * dt).min(1.0);

    for ev in motion.read() {
        cam.yaw   -= ev.delta.x * LOOK_SENS;
        cam.pitch  = (cam.pitch - ev.delta.y * LOOK_SENS)
            .clamp(-std::f32::consts::FRAC_PI_2 + 0.01,
                    std::f32::consts::FRAC_PI_2 - 0.01);
    }

    let fwd   = cam.forward();
    let right = cam.right();
    let up    = [0.0f32, 1.0, 0.0];
    let speed = MOVE_SPEED * dt;
    let mut mv = [0.0f32; 3];

    if keys.pressed(KeyCode::KeyW) || keys.pressed(KeyCode::ArrowUp) {
        for i in 0..3 { mv[i] += fwd[i] * speed; }
    }
    if keys.pressed(KeyCode::KeyS) || keys.pressed(KeyCode::ArrowDown) {
        for i in 0..3 { mv[i] -= fwd[i] * speed; }
    }
    if keys.pressed(KeyCode::KeyA) || keys.pressed(KeyCode::ArrowLeft) {
        for i in 0..3 { mv[i] -= right[i] * speed; }
    }
    if keys.pressed(KeyCode::KeyD) || keys.pressed(KeyCode::ArrowRight) {
        for i in 0..3 { mv[i] += right[i] * speed; }
    }
    if keys.pressed(KeyCode::KeyE) { for i in 0..3 { mv[i] += up[i] * speed; } }
    if keys.pressed(KeyCode::KeyQ) { for i in 0..3 { mv[i] -= up[i] * speed; } }

    for i in 0..3 { cam.position[i] += mv[i]; }

    update_tau_from_position(&mut cam);
}

/// §9.4 Follow-flow: bias camera velocity toward the highest-weight outgoing
/// neighbor of the nearest particle.  Call this from any system that holds
/// `ResMut<GraphCamera>` and has access to positions + CSR.
///
/// `modifier_held` — true when the user is holding the follow-flow modifier.
/// `positions`     — flat n×3 f32 particle positions (post-epoch, R_scene scale).
/// `csr`           — adjacency matrix for edge traversal.
pub fn apply_follow_flow(
    cam:          &mut GraphCamera,
    modifier_held: bool,
    positions:    &[f32],
    csr:          &crate::graph::Csr,
    dt:           f32,
) {
    if !modifier_held || positions.is_empty() || csr.n == 0 { return; }

    let n = positions.len() / 3;
    let cam_p = cam.position;

    // Find nearest particle.
    let nearest = (0..n).min_by(|&a, &b| {
        let da = dist2(cam_p, pos_of(positions, a));
        let db = dist2(cam_p, pos_of(positions, b));
        da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
    });
    let Some(near_idx) = nearest else { return };

    // Find highest-weight outgoing neighbor.
    let (cols, vals) = csr.row(near_idx);
    let best_nbr = cols.iter().zip(vals.iter())
        .max_by(|(_, &wa), (_, &wb)| wa.partial_cmp(&wb).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(&c, _)| c as usize);
    let Some(target_idx) = best_nbr else { return };

    // Bias camera velocity toward target particle.
    let target = pos_of(positions, target_idx);
    let flow_speed = MOVE_SPEED * dt;
    for i in 0..3 {
        let dir = target[i] - cam_p[i];
        cam.position[i] += dir.signum() * flow_speed.min(dir.abs() * 0.1);
    }
}

fn dist2(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = a[0]-b[0]; let dy = a[1]-b[1]; let dz = a[2]-b[2];
    dx*dx + dy*dy + dz*dz
}
fn pos_of(positions: &[f32], i: usize) -> [f32; 3] {
    let b = i * 3;
    [positions[b], positions[b+1], positions[b+2]]
}

/// §9.5: τ(p) = τ₀ · (1 + ‖p‖ / R_scene)^α (centroid ≈ origin after Procrustes).
fn update_tau_from_position(cam: &mut GraphCamera) {
    let p = cam.position;
    let dist = (p[0]*p[0] + p[1]*p[1] + p[2]*p[2]).sqrt();
    let tau_geo = TAU0 * (1.0 + dist / R_SCENE).powf(ALPHA);
    // Scroll-based tau_target takes precedence but cannot go below the geometric floor.
    cam.tau_target = cam.tau_target.max(tau_geo).clamp(TAU_MIN, TAU_MAX);
}

fn smooth_step(t: f32) -> f32 { t * t * (3.0 - 2.0 * t) }
fn lerp(a: f32, b: f32, t: f32) -> f32 { a + (b - a) * t }
fn lerp_angle(a: f32, b: f32, t: f32) -> f32 {
    let mut d = b - a;
    while d >  std::f32::consts::PI { d -= std::f32::consts::TAU; }
    while d < -std::f32::consts::PI { d += std::f32::consts::TAU; }
    a + d * t
}

/// §9.2 warp: initiate a 500 ms smooth-step camera fly toward `target_pos`,
/// oriented to look at `look_at`.
///
/// Call from any Bevy system that has `ResMut<GraphCamera>`.
pub fn initiate_warp(cam: &mut GraphCamera, target_pos: [f32; 3], look_at: [f32; 3]) {
    let dx = look_at[0] - target_pos[0];
    let dy = look_at[1] - target_pos[1];
    let dz = look_at[2] - target_pos[2];
    let horiz   = (dx * dx + dz * dz).sqrt();
    let to_yaw  = dz.atan2(dx) - std::f32::consts::FRAC_PI_2;
    let to_pitch = -(dy.atan2(horiz))
        .clamp(-std::f32::consts::FRAC_PI_2 + 0.01, std::f32::consts::FRAC_PI_2 - 0.01);

    cam.warp = Some(WarpAnim {
        from_pos:  cam.position,
        to_pos:    target_pos,
        to_yaw,
        to_pitch,
        elapsed:   0.0,
        duration:  0.5,
    });
}

// ---------------------------------------------------------------------------
// Unit tests (pure Rust, no Bevy app)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bevy::resources::GraphCamera;

    #[test]
    fn tau_at_origin_equals_tau0() {
        let mut cam = GraphCamera::default();
        cam.position = [0.0, 0.0, 0.0];
        cam.tau_target = TAU_MIN; // force to minimum
        update_tau_from_position(&mut cam);
        // At origin: dist=0, tau_geo = TAU0*(1+0)^2 = 0.1
        let expected = TAU0.clamp(TAU_MIN, TAU_MAX);
        assert!((cam.tau_target - expected).abs() < 1e-5,
            "tau_target={} expected={}", cam.tau_target, expected);
    }

    #[test]
    fn tau_increases_with_distance() {
        let mut cam_near = GraphCamera::default();
        cam_near.position = [0.0, 0.0, 0.0];
        cam_near.tau_target = TAU_MIN;
        update_tau_from_position(&mut cam_near);

        let mut cam_far = GraphCamera::default();
        cam_far.position = [0.0, 0.0, 1000.0]; // R_scene away
        cam_far.tau_target = TAU_MIN;
        update_tau_from_position(&mut cam_far);

        // τ at R_scene = τ₀ · (1+1)² = 0.4, greater than τ₀ = 0.1
        assert!(cam_far.tau_target > cam_near.tau_target,
            "tau should increase with distance: near={} far={}",
            cam_near.tau_target, cam_far.tau_target);
        assert!((cam_far.tau_target - 0.4).abs() < 0.01,
            "tau at R_scene={}", cam_far.tau_target);
    }

    #[test]
    fn warp_starts_and_completes() {
        let mut cam = GraphCamera::default();
        cam.position = [0.0, 0.0, 3000.0];

        initiate_warp(&mut cam, [0.0, 0.0, 30.0], [0.0, 0.0, 0.0]);
        assert!(cam.warp.is_some(), "warp should be active");

        // Simulate many frames until warp completes.
        for _ in 0..100 {
            if cam.warp.is_none() { break; }
            let (from_pos, to_pos, to_yaw, to_pitch, elapsed, duration) = {
                let w = cam.warp.as_mut().unwrap();
                w.elapsed = (w.elapsed + 0.02).min(w.duration);
                (w.from_pos, w.to_pos, w.to_yaw, w.to_pitch, w.elapsed, w.duration)
            };
            let t = smooth_step(elapsed / duration);
            for i in 0..3 { cam.position[i] = lerp(from_pos[i], to_pos[i], t); }
            let _ = (to_yaw, to_pitch);
            if elapsed >= duration { cam.warp = None; }
        }

        assert!(cam.warp.is_none(), "warp should have completed");
        assert!((cam.position[2] - 30.0).abs() < 1.0,
            "camera should be near target: z={}", cam.position[2]);
    }

    #[test]
    fn smooth_step_values() {
        assert_eq!(smooth_step(0.0), 0.0);
        assert_eq!(smooth_step(1.0), 1.0);
        assert!((smooth_step(0.5) - 0.5).abs() < 1e-6);
    }
}
