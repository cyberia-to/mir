//! Orbit camera: drag to rotate around origin, scroll to zoom.
//! §9.1 — §9.5 navigation.

use bevy::ecs::message::MessageReader;
use bevy::input::mouse::MouseWheel;
use bevy::prelude::*;
use super::resources::{GraphCamera, WarpAnim};

const TAU_MIN:   f32 = 0.01;
const TAU_MAX:   f32 = 100.0;
const LOOK_SENS: f32 = 0.005;
const DAMPING:   f32 = 8.0;

const TAU0:    f32 = 0.1;
const ALPHA:   f32 = 2.0;
const R_SCENE: f32 = 1000.0;

pub fn update_camera(
    mut cam:    ResMut<GraphCamera>,
    buttons:    Res<ButtonInput<MouseButton>>,
    keys:       Res<ButtonInput<KeyCode>>,
    mut cursor: MessageReader<CursorMoved>,
    mut scroll: MessageReader<MouseWheel>,
    time:       Res<Time>,
    windows:    Query<&Window>,
) {
    let dt = time.delta_secs();

    if let Ok(win) = windows.single() {
        cam.viewport = [win.width(), win.height()];
    }

    // Advance warp animation (§9.2).
    if cam.warp.is_some() {
        let (from_pos, to_pos, to_yaw, to_pitch, elapsed, duration) = {
            let w = cam.warp.as_mut().unwrap();
            w.elapsed = (w.elapsed + dt).min(w.duration);
            (w.from_pos, w.to_pos, w.to_yaw, w.to_pitch, w.elapsed, w.duration)
        };
        let t = smooth_step(elapsed / duration);
        for i in 0..3 { cam.position[i] = lerp(from_pos[i], to_pos[i], t); }
        let cur_yaw = cam.yaw; let cur_pitch = cam.pitch;
        cam.yaw   = lerp_angle(cur_yaw,   to_yaw,   t);
        cam.pitch = lerp(cur_pitch, to_pitch, t);
        if elapsed >= duration { cam.warp = None; }
        // Sync orbit_dist to the warp destination.
        let p = cam.position;
        cam.orbit_dist = (p[0]*p[0] + p[1]*p[1] + p[2]*p[2]).sqrt().max(10.0);
        update_tau_from_position(&mut cam);
        return;
    }

    // Scroll → smooth zoom. Clamp ev.y so a fast trackpad swipe doesn't jump.
    for ev in scroll.read() {
        let step = ev.y.clamp(-2.0, 2.0) * 0.06;
        cam.orbit_dist = (cam.orbit_dist * (-step).exp()).clamp(50.0, 30000.0);
    }

    // Drag (any button) → orbit: rotate yaw/pitch around the origin.
    let dragging = buttons.pressed(MouseButton::Left)
        || buttons.pressed(MouseButton::Right)
        || buttons.pressed(MouseButton::Middle);

    let mut had_cursor_event = false;
    for ev in cursor.read() {
        had_cursor_event = true;
        let cur = [ev.position.x, ev.position.y];
        if let Some(last) = cam.last_cursor {
            if dragging {
                let dx = cur[0] - last[0];
                let dy = cur[1] - last[1];
                cam.yaw   -= dx * LOOK_SENS;
                cam.pitch  = (cam.pitch + dy * LOOK_SENS)
                    .clamp(-std::f32::consts::FRAC_PI_2 + 0.01,
                            std::f32::consts::FRAC_PI_2 - 0.01);
            }
        }
        cam.last_cursor = Some(cur);
    }
    // Clear last_cursor when not dragging OR when no cursor events arrived this frame.
    // The second case handles mouseUp events consumed by the WebView overlay: if the
    // cursor enters a pointer-events region and the button release is swallowed, Bevy
    // sees no cursor events. Clearing here prevents a large delta jump when events resume.
    if !dragging || !had_cursor_event {
        cam.last_cursor = None;
    }

    // WASD + arrow keys: orbit rotate and zoom.
    let rot  = 1.5 * dt;
    let zoom = 1.0 + 2.0 * dt;
    if keys.pressed(KeyCode::KeyA) || keys.pressed(KeyCode::ArrowLeft)  { cam.yaw   -= rot; }
    if keys.pressed(KeyCode::KeyD) || keys.pressed(KeyCode::ArrowRight) { cam.yaw   += rot; }
    if keys.pressed(KeyCode::KeyW) || keys.pressed(KeyCode::ArrowUp)    { cam.pitch  = (cam.pitch - rot).clamp(-std::f32::consts::FRAC_PI_2 + 0.01, std::f32::consts::FRAC_PI_2 - 0.01); }
    if keys.pressed(KeyCode::KeyS) || keys.pressed(KeyCode::ArrowDown)  { cam.pitch  = (cam.pitch + rot).clamp(-std::f32::consts::FRAC_PI_2 + 0.01, std::f32::consts::FRAC_PI_2 - 0.01); }
    if keys.pressed(KeyCode::KeyQ) || keys.pressed(KeyCode::Minus)      { cam.orbit_dist = (cam.orbit_dist * zoom).clamp(50.0, 30000.0); }
    if keys.pressed(KeyCode::KeyE) || keys.pressed(KeyCode::Equal)      { cam.orbit_dist = (cam.orbit_dist / zoom).clamp(50.0, 30000.0); }

    // Recompute position from spherical orbit coordinates (orbit around origin).
    // forward() = [cp*sy, sp, -cp*cy]
    // position  = -dist * forward = [-dist*cp*sy, -dist*sp, dist*cp*cy]
    let (sy, cy) = cam.yaw.sin_cos();
    let (sp, cp) = cam.pitch.sin_cos();
    let d = cam.orbit_dist;
    cam.position = [-d * cp * sy, -d * sp, d * cp * cy];

    let gap = cam.tau_target - cam.tau;
    cam.tau += gap * (DAMPING * dt).min(1.0);
    update_tau_from_position(&mut cam);
}

/// §9.4 Follow-flow.
pub fn apply_follow_flow(
    cam:           &mut GraphCamera,
    modifier_held: bool,
    positions:     &[f32],
    csr:           &crate::graph::Csr,
    _dt:           f32,
) {
    if !modifier_held || positions.is_empty() || csr.n == 0 { return; }
    let n = positions.len() / 3;
    let cam_p = cam.position;
    let nearest = (0..n).min_by(|&a, &b| {
        let da = dist2(cam_p, pos_of(positions, a));
        let db = dist2(cam_p, pos_of(positions, b));
        da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
    });
    let Some(near_idx) = nearest else { return };
    let (cols, vals) = csr.row(near_idx);
    let best = cols.iter().zip(vals.iter())
        .max_by(|(_, &wa), (_, &wb)| wa.partial_cmp(&wb).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(&c, _)| c as usize);
    let Some(target_idx) = best else { return };
    let target = pos_of(positions, target_idx);
    // Bias orbit toward target by nudging yaw/pitch.
    let dx = target[0] - cam_p[0];
    let dz = target[2] - cam_p[2];
    cam.yaw += dz.atan2(dx) * 0.001;
}

fn dist2(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = a[0]-b[0]; let dy = a[1]-b[1]; let dz = a[2]-b[2];
    dx*dx + dy*dy + dz*dz
}
fn pos_of(positions: &[f32], i: usize) -> [f32; 3] {
    let b = i * 3;
    [positions[b], positions[b+1], positions[b+2]]
}

fn update_tau_from_position(cam: &mut GraphCamera) {
    let p = cam.position;
    let dist = (p[0]*p[0] + p[1]*p[1] + p[2]*p[2]).sqrt();
    let tau_geo = TAU0 * (1.0 + dist / R_SCENE).powf(ALPHA);
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

pub fn initiate_warp(cam: &mut GraphCamera, target_pos: [f32; 3], look_at: [f32; 3]) {
    let dx = look_at[0] - target_pos[0];
    let dy = look_at[1] - target_pos[1];
    let dz = look_at[2] - target_pos[2];
    let horiz   = (dx * dx + dz * dz).sqrt();
    let to_yaw  = dz.atan2(dx) - std::f32::consts::FRAC_PI_2;
    let to_pitch = -(dy.atan2(horiz))
        .clamp(-std::f32::consts::FRAC_PI_2 + 0.01, std::f32::consts::FRAC_PI_2 - 0.01);
    cam.warp = Some(WarpAnim {
        from_pos: cam.position,
        to_pos:   target_pos,
        to_yaw, to_pitch,
        elapsed:  0.0,
        duration: 0.5,
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bevy::resources::GraphCamera;

    #[test]
    fn tau_at_origin_equals_tau0() {
        let mut cam = GraphCamera::default();
        cam.position = [0.0, 0.0, 0.0];
        cam.tau_target = TAU_MIN;
        update_tau_from_position(&mut cam);
        let expected = TAU0.clamp(TAU_MIN, TAU_MAX);
        assert!((cam.tau_target - expected).abs() < 1e-5);
    }

    #[test]
    fn tau_increases_with_distance() {
        let mut cam_near = GraphCamera::default();
        cam_near.position = [0.0, 0.0, 0.0];
        cam_near.tau_target = TAU_MIN;
        update_tau_from_position(&mut cam_near);

        let mut cam_far = GraphCamera::default();
        cam_far.position = [0.0, 0.0, 1000.0];
        cam_far.tau_target = TAU_MIN;
        update_tau_from_position(&mut cam_far);

        assert!(cam_far.tau_target > cam_near.tau_target);
        assert!((cam_far.tau_target - 0.4).abs() < 0.01);
    }

    #[test]
    fn warp_starts_and_completes() {
        let mut cam = GraphCamera::default();
        cam.position = [0.0, 0.0, 3000.0];
        initiate_warp(&mut cam, [0.0, 0.0, 30.0], [0.0, 0.0, 0.0]);
        assert!(cam.warp.is_some());
        for _ in 0..100 {
            if cam.warp.is_none() { break; }
            let (from_pos, to_pos, _, _, elapsed, duration) = {
                let w = cam.warp.as_mut().unwrap();
                w.elapsed = (w.elapsed + 0.02).min(w.duration);
                (w.from_pos, w.to_pos, w.to_yaw, w.to_pitch, w.elapsed, w.duration)
            };
            let t = smooth_step(elapsed / duration);
            for i in 0..3 { cam.position[i] = lerp(from_pos[i], to_pos[i], t); }
            if elapsed >= duration { cam.warp = None; }
        }
        assert!(cam.warp.is_none());
        assert!((cam.position[2] - 30.0).abs() < 1.0);
    }

    #[test]
    fn smooth_step_values() {
        assert_eq!(smooth_step(0.0), 0.0);
        assert_eq!(smooth_step(1.0), 1.0);
        assert!((smooth_step(0.5) - 0.5).abs() < 1e-6);
    }
}
