//! Bevy plugin for the mir render engine.
//! Wires the graph world into the Bevy WorldState FSM.
//!
//! Feature-gated: compile with `--features bevy-plugin`.
//!
//! NOTE: WorldState::Graph must be added to the cyb/bevy WorldState enum
//! (see /Users/master/cyber/cyb/bevy/src/worlds/mod.rs — TODO).

pub mod components;
pub mod resources;
pub mod camera;
pub mod world;
pub mod plugin;

pub use plugin::GraphWorldPlugin;
