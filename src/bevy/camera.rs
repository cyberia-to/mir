//! Camera τ (heat-kernel scale) smooth-step from scroll input.
//! Scroll up → smaller τ (zoom in, finer spectral detail).
//! Scroll down → larger τ (zoom out, coarser clustering).

use bevy::ecs::message::MessageReader;
use bevy::input::mouse::MouseWheel;
use bevy::prelude::*;

use super::resources::GraphCamera;

/// Scroll sensitivity: fraction of tau_target changed per scroll unit.
const SCROLL_SENSITIVITY: f32 = 0.1;
/// τ clamp range: [τ_min, τ_max].
const TAU_MIN: f32 = 0.01;
const TAU_MAX: f32 = 100.0;
/// Smooth-step damping coefficient (fraction of gap closed per second).
const DAMPING: f32 = 8.0;

/// System: read scroll messages → adjust tau_target, smooth-step tau toward it.
pub fn update_camera_tau(
    mut cam: ResMut<GraphCamera>,
    mut scroll: MessageReader<MouseWheel>,
    time: Res<Time>,
) {
    // Accumulate scroll delta this frame.
    let mut delta = 0.0f32;
    for ev in scroll.read() {
        // y > 0 = scroll up → zoom in → smaller τ.
        delta += ev.y;
    }

    if delta != 0.0 {
        // Multiplicative adjustment so zoom feels linear on a log scale.
        let factor = 1.0 - delta * SCROLL_SENSITIVITY;
        cam.tau_target = (cam.tau_target * factor).clamp(TAU_MIN, TAU_MAX);
    }

    // Exponential smooth-step: move tau toward tau_target each frame.
    let dt = time.delta_secs();
    let gap = cam.tau_target - cam.tau;
    cam.tau += gap * (DAMPING * dt).min(1.0);
}
