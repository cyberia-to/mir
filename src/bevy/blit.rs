//! The painted frame reaches the screen without passing through the CPU.
//!
//! The frame used to travel: paint buffer → staging buffer → mapped read →
//! `Vec<u8>` → `Image::data` → Bevy re-uploads the whole texture. At a phone's
//! full resolution that is ten megabytes making a round trip every frame, and
//! it measured at 21 ms of a 40 ms frame — more than four times everything mir
//! itself does. Removing just the upload took the Pixel from 25 to 54 fps.
//!
//! mir already renders on Bevy's own device and queue (see `install_shared`),
//! so the buffer and the texture live in the same place and the copy is one
//! command. Queue order does the synchronising: the blit is submitted after
//! the paint dispatch, on the same queue, so it cannot read a half-drawn frame.
//!
//! Apple keeps the CPU path. There mir renders on aruminium's Metal device,
//! which is not Bevy's, so there is no shared texture to copy into — and at
//! 140 fps there is nothing to win.

use std::sync::{Arc, Mutex};

use bevy::prelude::*;
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::render_asset::RenderAssets;
use bevy::render::renderer::{RenderDevice, RenderQueue};
use bevy::render::texture::GpuImage;
use bevy::render::{Render, RenderApp, RenderSystems};

/// What the render world needs to put this frame on the screen.
#[derive(Clone)]
pub struct FrameCopy {
    pub buffer: wgpu::Buffer,
    pub image:  AssetId<Image>,
    pub width:  u32,
    pub height: u32,
}

/// The main world writes the latest frame here; the render world copies it.
///
/// A mutex rather than a channel: there is nothing to queue. Only the newest
/// frame is ever worth drawing, and a render world that misses one simply
/// draws the same texture again.
#[derive(Resource, Clone, ExtractResource)]
pub struct FrameHandoff(pub Arc<Mutex<Option<FrameCopy>>>);

impl Default for FrameHandoff {
    fn default() -> Self {
        Self(Arc::new(Mutex::new(None)))
    }
}

impl FrameHandoff {
    pub fn publish(&self, copy: FrameCopy) {
        if let Ok(mut slot) = self.0.lock() {
            *slot = Some(copy);
        }
    }
}

pub struct FrameBlitPlugin;

impl Plugin for FrameBlitPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<FrameHandoff>()
            .add_plugins(ExtractResourcePlugin::<FrameHandoff>::default());

        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            // After PrepareAssets, so the texture for a freshly created or
            // resized image exists; before the passes that sample it.
            render_app.add_systems(Render, blit_frame.in_set(RenderSystems::Prepare));
        }
    }
}

fn blit_frame(
    handoff: Res<FrameHandoff>,
    images:  Res<RenderAssets<GpuImage>>,
    device:  Res<RenderDevice>,
    queue:   Res<RenderQueue>,
) {
    let job = match handoff.0.lock() {
        Ok(slot) => slot.clone(),
        Err(_) => return,
    };
    let Some(job) = job else { return };
    let Some(target) = images.get(job.image) else { return };

    // A resize lands in the two worlds a frame apart. Copying a frame of one
    // size into a texture of another is a validation error and, on a phone,
    // the end of the process — so when they disagree, skip a frame instead.
    if target.size.width != job.width || target.size.height != job.height {
        return;
    }

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("mir-frame-blit"),
    });
    enc.copy_buffer_to_texture(
        wgpu::TexelCopyBufferInfo {
            buffer: &job.buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                // Rows must be a multiple of 256 bytes, which is why the frame
                // is sized to a multiple of 64 pixels. See `render_size`.
                bytes_per_row: Some(job.width * 4),
                rows_per_image: Some(job.height),
            },
        },
        wgpu::TexelCopyTextureInfo {
            texture:   &target.texture,
            mip_level: 0,
            origin:    wgpu::Origin3d::ZERO,
            aspect:    wgpu::TextureAspect::All,
        },
        wgpu::Extent3d {
            width:  job.width,
            height: job.height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit([enc.finish()]);
}
