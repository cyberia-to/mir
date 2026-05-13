//! GraphWorldPlugin wires all mir graph-world systems into a Bevy App.

use bevy::prelude::*;
use super::camera::update_camera;
use super::resources::WarpTarget;
use super::world::{
    GraphWorldState,
    animate_edges, composite, dispatch_tiers,
    on_enter_graph, on_exit_graph,
    swap_epoch_if_ready, sync_visible_entities, tick_diffusion,
    follow_flow_system, warp_to_system,
};

pub struct GraphWorldPlugin;

impl Plugin for GraphWorldPlugin {
    fn build(&self, app: &mut App) {
        app
            .init_state::<GraphWorldState>()
            .init_resource::<WarpTarget>()
            .add_systems(OnEnter(GraphWorldState::Active), on_enter_graph)
            .add_systems(OnExit(GraphWorldState::Active),  on_exit_graph)
            .add_systems(PreUpdate,
                swap_epoch_if_ready.run_if(in_state(GraphWorldState::Active)))
            .add_systems(Update,
                (tick_diffusion, warp_to_system, update_camera, follow_flow_system, sync_visible_entities)
                    .chain()
                    .run_if(in_state(GraphWorldState::Active)))
            .add_systems(PostUpdate,
                (dispatch_tiers, animate_edges, composite)
                    .chain()
                    .run_if(in_state(GraphWorldState::Active)));
    }
}
