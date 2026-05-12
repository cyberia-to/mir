//! T0 content entry: camera transition FSM + sandbox. Step 10.
//!
//! R-1.0 §6.1: T0 fires when screen-pixel diameter ≥ S_T0 = 200 px.
//! The camera animates into the particle's interior over 500 ms (smooth-step).
//! §9.3: inside the particle a sandboxed content surface is revealed.

/// Camera command emitted by the T0 FSM each frame.
#[derive(Debug, Clone)]
pub enum CameraCommand {
    /// Camera is approaching the particle; `t` ∈ [0, 1).
    Approaching { particle: u32, t: f32 },
    /// Camera is fully inside the particle.
    Inside { particle: u32 },
    /// Camera is exiting; `t` ∈ [0, 1).
    Exiting { particle: u32, t: f32 },
}

/// Camera FSM state for T0 content entry.
#[derive(Default, Clone, Debug)]
pub enum T0State {
    /// No particle selected; camera in free-fly graph mode.
    #[default]
    Outside,
    /// Animating toward `target` particle; `progress` ∈ [0, 1].
    Approaching { target: u32, progress: f32 },
    /// Camera is inside `particle` — content sandbox is active.
    Inside { particle: u32 },
    /// Animating back to graph; `progress` ∈ [0, 1].
    Exiting { particle: u32, progress: f32 },
}

/// Transition duration in seconds (§9.3 specifies 500 ms).
const TRANSITION_SECS: f32 = 0.5;

/// T0 camera FSM.
pub struct T0Pass {
    pub state: T0State,
}

impl T0Pass {
    pub fn new() -> Self {
        Self { state: T0State::Outside }
    }

    /// Begin approach transition toward `particle_idx`.
    /// Idempotent if already approaching or inside that particle.
    pub fn enter(&mut self, particle_idx: u32) {
        match self.state {
            T0State::Approaching { target, .. } if target == particle_idx => {}
            T0State::Inside { particle } if particle == particle_idx => {}
            _ => {
                self.state = T0State::Approaching {
                    target:   particle_idx,
                    progress: 0.0,
                };
            }
        }
    }

    /// Begin exit transition from the current particle back to graph view.
    pub fn exit(&mut self) {
        if let T0State::Inside { particle } = self.state {
            self.state = T0State::Exiting { particle, progress: 0.0 };
        }
    }

    /// Advance the FSM by `dt` seconds and return a `CameraCommand` if active.
    ///
    /// `positions` — flat f32 array, stride 3 (x, y, z per particle).
    /// `radii`     — per-particle radius in world-space units.
    pub fn update(
        &mut self,
        dt:        f32,
        positions: &[f32],
        radii:     &[f32],
    ) -> Option<CameraCommand> {
        match &mut self.state {
            T0State::Outside => None,

            T0State::Approaching { target, progress } => {
                *progress = (*progress + dt / TRANSITION_SECS).min(1.0);
                let t = smooth_step(*progress);

                // Compute entry position: particle center + radius along -Z.
                let p = entry_pos(*target, positions, radii);
                let _ = p; // consumed by the real camera move; stub here

                let cmd = CameraCommand::Approaching { particle: *target, t };

                if *progress >= 1.0 {
                    let particle = *target;
                    self.state = T0State::Inside { particle };
                }

                Some(cmd)
            }

            T0State::Inside { particle } => {
                Some(CameraCommand::Inside { particle: *particle })
            }

            T0State::Exiting { particle, progress } => {
                *progress = (*progress + dt / TRANSITION_SECS).min(1.0);
                let t = smooth_step(*progress);
                let particle = *particle;

                let cmd = CameraCommand::Exiting { particle, t };

                if *progress >= 1.0 {
                    self.state = T0State::Outside;
                }

                Some(cmd)
            }
        }
    }
}

impl Default for T0Pass {
    fn default() -> Self { Self::new() }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Cubic smooth-step: maps t ∈ [0, 1] → [0, 1] with zero derivative at ends.
pub fn smooth_step(t: f32) -> f32 {
    t * t * (3.0 - 2.0 * t)
}

/// World-space position of the camera entry point for `particle_idx`.
/// The camera lands just inside the surface: center + radius * 0.9 along -Z.
fn entry_pos(particle_idx: u32, positions: &[f32], radii: &[f32]) -> [f32; 3] {
    let i = particle_idx as usize;
    let base = i * 3;
    if base + 2 >= positions.len() || i >= radii.len() {
        return [0.0, 0.0, 0.0];
    }
    let r = radii[i] * 0.9;
    [positions[base], positions[base + 1], positions[base + 2] + r]
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn approach_completes_to_inside() {
        let mut pass = T0Pass::new();
        pass.enter(0);

        let positions = [0.0f32, 0.0, 0.0];
        let radii     = [10.0f32];

        // Simulate enough frames to complete the transition.
        let mut last_cmd = None;
        for _ in 0..100 {
            last_cmd = pass.update(0.02, &positions, &radii);
        }

        // Should be Inside after transition completes.
        match pass.state {
            T0State::Inside { particle } => assert_eq!(particle, 0),
            other => panic!("expected Inside, got {:?}", other),
        }
        // Last command while approaching is Approaching.
        // After finishing, update returns Inside.
        match last_cmd {
            Some(CameraCommand::Inside { particle: 0 }) => {}
            Some(other) => panic!("unexpected command: {:?}", other),
            None => panic!("no command"),
        }
    }

    #[test]
    fn exit_returns_to_outside() {
        let mut pass = T0Pass::new();
        pass.state = T0State::Inside { particle: 1 };
        pass.exit();

        let positions = [0.0f32, 0.0, 0.0, 1.0, 2.0, 3.0];
        let radii     = [5.0f32, 5.0];

        for _ in 0..100 {
            pass.update(0.02, &positions, &radii);
        }
        assert!(matches!(pass.state, T0State::Outside));
    }

    #[test]
    fn smooth_step_endpoints() {
        assert_eq!(smooth_step(0.0), 0.0);
        assert_eq!(smooth_step(1.0), 1.0);
        assert!((smooth_step(0.5) - 0.5).abs() < 1e-6);
    }
}
