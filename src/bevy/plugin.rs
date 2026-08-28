//! GraphWorldPlugin wires all mir graph-world systems into a Bevy App.

use bevy::prelude::*;
use super::camera::update_camera;
use super::resources::WarpTarget;
use super::world::{
    GraphWorldState,
    animate_edges, dispatch_tiers,
    on_enter_graph, on_exit_graph,
    swap_epoch_if_ready, sync_visible_entities, tick_diffusion, track_frame_size,
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
                    .run_if(in_state(GraphWorldState::Active)));

        // The last step of a frame is getting what was painted onto the
        // screen, and how that is done depends on whether mir renders on
        // Bevy's own GPU device. On Apple it does not — mir is on aruminium's
        // Metal device — so the frame is read back and handed over as image
        // data. Everywhere else the two share a device and a queue, and the
        // render world copies buffer to texture without the CPU seeing it.
        #[cfg(target_vendor = "apple")]
        app.add_systems(PostUpdate,
            (track_frame_size, dispatch_tiers, animate_edges, super::world::composite)
                .chain()
                .run_if(in_state(GraphWorldState::Active)));

        #[cfg(not(target_vendor = "apple"))]
        {
            app.add_systems(PostUpdate,
                (track_frame_size, dispatch_tiers, animate_edges, super::world::publish_frame)
                    .chain()
                    .run_if(in_state(GraphWorldState::Active)));
            app.add_plugins(super::blit::FrameBlitPlugin);
        }
    }
}
