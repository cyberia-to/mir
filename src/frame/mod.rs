//! Frame work: display-refresh foreground thread.
//! cull → tier dispatch → diffusion → edges → composite.
//!
//! All GPU work goes through aruminium. Steps 4–10.

pub mod cull;
pub mod diffusion;
pub mod edges;
pub mod paint;
