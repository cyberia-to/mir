//! GraphWorldPlugin — wires all mir graph-world systems into a Bevy App.
//!
//! Standalone usage (mir as its own app):
//! ```no_run
//! use bevy::prelude::*;
//! use mir::bevy::{GraphWorldPlugin, world::GraphWorldState};
//!
//! App::new()
//!     .add_plugins(DefaultPlugins)
//!     .add_plugins(GraphWorldPlugin)
//!     .run();
//! ```
//!
//! TODO: When integrating into cyb/bevy, replace `GraphWorldState` with
//! `WorldState::Graph` from `cyb_bevy::worlds`.

use bevy::prelude::*;

use super::camera::update_camera_tau;
use super::world::{
    GraphWorldState,
    animate_edges, composite, dispatch_tiers,
    on_enter_graph, on_exit_graph,
    swap_epoch_if_ready, sync_visible_entities, tick_diffusion,
};

pub struct GraphWorldPlugin;

impl Plugin for GraphWorldPlugin {
    fn build(&self, app: &mut App) {
        app
            // Register the stand-in state (remove when using WorldState::Graph).
            .init_state::<GraphWorldState>()

            // Entry / exit.
            .add_systems(OnEnter(GraphWorldState::Active), on_enter_graph)
            .add_systems(OnExit(GraphWorldState::Active),  on_exit_graph)

            // PreUpdate: swap double-buffer if background epoch thread finished.
            .add_systems(
                PreUpdate,
                swap_epoch_if_ready.run_if(in_state(GraphWorldState::Active)),
            )

            // Update: diffusion tick → camera τ → entity sync (ordered).
            .add_systems(
                Update,
                (tick_diffusion, update_camera_tau, sync_visible_entities)
                    .chain()
                    .run_if(in_state(GraphWorldState::Active)),
            )

            // PostUpdate: tier dispatch → edge animation → composite (ordered).
            .add_systems(
                PostUpdate,
                (dispatch_tiers, animate_edges, composite)
                    .chain()
                    .run_if(in_state(GraphWorldState::Active)),
            );
    }
}
