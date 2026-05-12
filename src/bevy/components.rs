//! Bevy ECS components for visible graph particles.

use bevy::prelude::*;

/// Marks a Bevy entity as a visible particle in the graph world.
/// The inner value is the particle index into EpochState::positions.
#[derive(Component)]
pub struct VisibleParticle(pub u32);

/// Tier level for this particle's LOD rendering.
/// 0 = T0 (content entry), 1 = T1 (surface+label), 2 = T2 (sphere),
/// 3 = T3 (point), 4 = T∞ (impostor / culled-to-cluster).
#[derive(Component)]
pub struct TierLevel(pub u8);
