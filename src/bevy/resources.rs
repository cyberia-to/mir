//! Bevy ECS resources for the graph world.

use std::sync::{Arc, RwLock};
use bevy::prelude::*;

/// Shared epoch state, written by the background EpochWorker thread
/// and read (swapped) by the frame thread each time a new epoch completes.
#[derive(Resource)]
pub struct EpochStateRes {
    pub inner: Arc<RwLock<Option<crate::epoch::EpochState>>>,
}

/// Camera parameters for the heat-kernel τ zoom.
/// Scroll input moves tau_target; tau tracks it with smooth-step damping.
#[derive(Resource)]
pub struct GraphCamera {
    /// Current heat-kernel scale (controls spectral zoom level).
    pub tau:        f32,
    /// Smooth-step target; set by scroll input.
    pub tau_target: f32,
}

impl Default for GraphCamera {
    fn default() -> Self {
        Self { tau: 1.0, tau_target: 1.0 }
    }
}

/// GPU buffer metadata — positions, focus, BVH are uploaded once per epoch swap.
/// The actual Metal buffers live in the aruminium Gpu handle (not a Bevy resource).
#[derive(Resource, Default)]
pub struct GpuBuffers {
    /// Current particle count reflected in GPU buffers.
    pub n_particles: usize,
}
