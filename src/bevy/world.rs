//! Graph world systems: OnEnter / Update / OnExit for the graph render world.
//!
//! TODO: When integrating into cyb/bevy, replace `GraphWorldState::Active` with
//! `WorldState::Graph` from `cyb_bevy::worlds::WorldState` and remove the local
//! `GraphWorldState` definition below.

use std::sync::{Arc, RwLock};
use bevy::prelude::*;

use super::components::{TierLevel, VisibleParticle};
use super::resources::{EpochStateRes, GpuBuffers, GraphCamera};

// ---------------------------------------------------------------------------
// Local stand-in for WorldState::Graph.
// Replace with cyb_bevy::worlds::WorldState::Graph when integrating into cyb.
// ---------------------------------------------------------------------------

/// Minimal FSM state used by the standalone mir plugin.
/// When embedding in cyb/bevy, wire OnEnter/OnExit to WorldState::Graph instead.
#[derive(States, Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum GraphWorldState {
    #[default]
    Inactive,
    /// Equivalent to WorldState::Graph in the cyb/bevy shell.
    Active,
}

// ---------------------------------------------------------------------------
// Marker component for the loading overlay UI text entity.
// ---------------------------------------------------------------------------

#[derive(Component)]
pub struct LoadingOverlay;

// ---------------------------------------------------------------------------
// OnEnter(GraphWorldState::Active)
// ---------------------------------------------------------------------------

pub fn on_enter_graph(mut commands: Commands) {
    info!("mir: entering graph world");

    // 1. Spawn loading overlay.
    commands.spawn((
        LoadingOverlay,
        Text::new("loading graph…"),
        TextFont {
            font_size: 32.0,
            ..default()
        },
        TextColor(Color::WHITE),
        Node {
            position_type: PositionType::Absolute,
            left:   Val::Px(20.0),
            bottom: Val::Px(20.0),
            ..default()
        },
    ));

    // 2. Insert epoch / GPU resources.
    //    Phase 1: EpochState starts as None; background thread fills it.
    //    TODO: spawn EpochWorker thread here, pointing at the .graph mmap path.
    let epoch_inner: Arc<RwLock<Option<crate::epoch::EpochState>>> =
        Arc::new(RwLock::new(None));

    commands.insert_resource(EpochStateRes { inner: epoch_inner });
    commands.insert_resource(GraphCamera::default());
    commands.insert_resource(GpuBuffers::default());
}

// ---------------------------------------------------------------------------
// PreUpdate: swap epoch if the background thread produced a new one.
// ---------------------------------------------------------------------------

pub fn swap_epoch_if_ready(
    mut gpu: ResMut<GpuBuffers>,
    epoch_res: Res<EpochStateRes>,
    loading_q: Query<Entity, With<LoadingOverlay>>,
    mut commands: Commands,
) {
    // Try to take a completed EpochState out of the Arc<RwLock<Option<…>>>.
    let mut lock = match epoch_res.inner.try_write() {
        Ok(l) => l,
        Err(_) => return, // writer still active; try next frame
    };

    if let Some(epoch) = lock.take() {
        // Upload to GPU (stub — real upload via aruminium follows in step 4).
        gpu.n_particles = epoch.positions.len() / 3;
        info!("mir: epoch swapped, {} particles", gpu.n_particles);

        // Despawn loading overlay now that the first epoch is ready.
        for entity in loading_q.iter() {
            commands.entity(entity).despawn();
        }

        // Put the epoch back so other systems can read it this frame.
        // (In step 4 the GPU buffers are the source of truth; CPU data is dropped.)
        *lock = Some(epoch);
    }
}

// ---------------------------------------------------------------------------
// Update: per-frame systems (stubs; real implementations in steps 4–9).
// ---------------------------------------------------------------------------

pub fn tick_diffusion(
    _gpu: Res<GpuBuffers>,
    _time: Res<Time>,
) {
    // TODO step 4: dispatch acpu diffusion compute shader for 1–2 PageRank steps.
}

pub fn sync_visible_entities(
    _gpu: Res<GpuBuffers>,
    _commands: Commands,
) {
    // TODO step 4: read visible_out buffer from cull pass, reconcile ECS entities.
    // Spawn VisibleParticle + TierLevel for new particles, despawn removed ones.
    let _ = (VisibleParticle(0), TierLevel(0)); // suppress unused-import warning
}

// ---------------------------------------------------------------------------
// PostUpdate stubs
// ---------------------------------------------------------------------------

pub fn dispatch_tiers(_gpu: Res<GpuBuffers>) {
    // TODO step 5–9: call CullPass::run, then T3/T2/T1/T0 passes.
}

pub fn animate_edges(_gpu: Res<GpuBuffers>, _time: Res<Time>) {
    // TODO step 9: update EdgePass flow UVs and dispatch tube geometry compute.
}

pub fn composite(_gpu: Res<GpuBuffers>) {
    // TODO step 7+: IOSurface composite via unimem zero-copy.
}

// ---------------------------------------------------------------------------
// OnExit(GraphWorldState::Active)
// ---------------------------------------------------------------------------

pub fn on_exit_graph(
    mut commands: Commands,
    particles_q: Query<Entity, With<VisibleParticle>>,
    loading_q:   Query<Entity, With<LoadingOverlay>>,
) {
    info!("mir: exiting graph world");

    // Despawn all visible particle entities.
    for entity in particles_q.iter() {
        commands.entity(entity).despawn();
    }
    // Despawn any lingering loading overlay.
    for entity in loading_q.iter() {
        commands.entity(entity).despawn();
    }

    // NOTE: EpochStateRes is intentionally retained across exit
    // because eigenvector computation is expensive and the next
    // WorldState::Graph entry can reuse the existing layout.
    // GpuBuffers and GraphCamera are similarly retained.
}
