//! Graph world systems.

use std::sync::{Arc, RwLock};
use bevy::asset::RenderAssetUsages;
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages};

use crate::epoch::EpochWorker;
use crate::frame::cull::TierLevel;
use crate::frame::diffusion::diffusion_step;

use super::components::{TierLevel as CompTier, VisibleParticle};
use super::resources::{EpochStateRes, GpuBuffers, GraphCamera, GraphWorldConfig, WarpTarget};

#[derive(States, Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum GraphWorldState { #[default] Inactive, Active }

#[derive(Component)] pub struct LoadingOverlay;
#[derive(Component)] pub struct RenderOutput;

// ── OnEnter ─────────────────────────────────────────────────────────────────

pub fn on_enter_graph(
    mut commands: Commands,
    mut images:   ResMut<Assets<Image>>,
    config:       Option<Res<GraphWorldConfig>>,
) {
    info!("mir: entering graph world");
    let w = 1280u32; let h = 720u32;

    // Create blank RGBA8 output image.
    let mut image = Image::new(
        Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        TextureDimension::D2,
        vec![20u8; (w * h * 4) as usize],
        TextureFormat::Rgba8Unorm,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    );
    image.texture_descriptor.usage =
        TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST;
    let img_handle = images.add(image);

    // Fullscreen render output (behind other UI).
    commands.spawn((
        RenderOutput,
        ImageNode { image: img_handle.clone(), ..default() },
        Node {
            width: Val::Percent(100.0),
            height: Val::Percent(100.0),
            position_type: PositionType::Absolute,
            ..default()
        },
        ZIndex(-1),
    ));

    // Loading overlay.
    commands.spawn((
        LoadingOverlay,
        Text::new("loading graph\u{2026}"),
        TextFont { font_size: 28.0, ..default() },
        TextColor(Color::WHITE),
        Node {
            position_type: PositionType::Absolute,
            left: Val::Px(20.0), bottom: Val::Px(20.0),
            ..default()
        },
    ));

    let mut gpu = GpuBuffers::new();
    gpu.viewport = [w, h];
    gpu.output_image = Some(img_handle);

    let epoch_arc: Arc<RwLock<Option<crate::epoch::EpochState>>> =
        Arc::new(RwLock::new(None));

    if let Some(cfg) = config {
        let vocab = Arc::new(crate::graph::ParticleIndex::empty());
        gpu.csr = Some(Arc::clone(&cfg.graph));
        let (_worker, state) = EpochWorker::spawn(Arc::clone(&cfg.graph), vocab);
        commands.insert_resource(EpochStateRes { inner: state });
    } else {
        commands.insert_resource(EpochStateRes { inner: epoch_arc });
    }

    commands.insert_resource(GraphCamera::default());
    commands.insert_resource(gpu);
}

// ── PreUpdate ────────────────────────────────────────────────────────────────

pub fn swap_epoch_if_ready(
    mut gpu:      ResMut<GpuBuffers>,
    epoch_res:    Res<EpochStateRes>,
    loading_q:    Query<Entity, With<LoadingOverlay>>,
    mut commands: Commands,
) {
    // Only upload once: once gpu has particles, skip re-upload.
    if gpu.n_particles > 0 { return; }
    let mut lock = match epoch_res.inner.try_write() { Ok(l) => l, Err(_) => return };
    if let Some(epoch) = lock.as_ref() {
        info!("mir: epoch ready, {} particles", epoch.positions.len() / 3);
        gpu.upload_epoch(epoch);
        for e in loading_q.iter() { commands.entity(e).despawn(); }
    }
}

// ── Update ───────────────────────────────────────────────────────────────────

pub fn tick_diffusion(mut gpu: ResMut<GpuBuffers>) {
    if gpu.n_particles == 0 { return }
    let Some(csr) = gpu.csr.clone() else { return };
    let d_inv = gpu.d_inv.clone();
    diffusion_step(&csr, &d_inv, &mut gpu.focus);
}

pub fn sync_visible_entities(
    mut gpu:      ResMut<GpuBuffers>,
    cam:          Res<GraphCamera>,
    mut commands: Commands,
    old_q:        Query<Entity, With<VisibleParticle>>,
) {
    if gpu.n_particles == 0 { return }
    let camera = cam.to_gpu_camera();

    // Call CullPass with BVH buffer (or dummy if BVH not yet uploaded).
    let visible = if let (Some(cull), Some(pb), Some(rb)) =
        (&gpu.cull, &gpu.pos_buf, &gpu.rad_buf)
    {
        let bvh_ref = gpu.bvh_buf.as_ref().or(gpu.dummy_buf.as_ref());
        let Some(bb) = bvh_ref else { return };
        match cull.run(pb, rb, bb, &camera, gpu.n_particles as u32) {
            Ok(vs) => vs.entries,
            Err(e) => { warn!("cull: {e}"); return; }
        }
    } else { return };

    for e in old_q.iter() { commands.entity(e).despawn(); }
    for &(idx, tier) in &visible {
        commands.spawn((VisibleParticle(idx), CompTier(tier as u8)));
    }
    {
        static ONCE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if gpu.visible.len() != visible.len()
            || !ONCE.swap(true, std::sync::atomic::Ordering::Relaxed)
        {
            debug!("mir: cull -> {} visible of {}", visible.len(), gpu.n_particles);
        }
    }
    gpu.visible = visible;
}

// ── PostUpdate ────────────────────────────────────────────────────────────────

pub fn dispatch_tiers(
    mut gpu: ResMut<GpuBuffers>,
    cam:     Res<GraphCamera>,
) {
    if gpu.visible.is_empty() { return }
    let camera = cam.to_gpu_camera();
    let [w, h] = gpu.viewport;

    // Read positions for CPU depth sort.
    trace_step("read positions");
    let positions: Vec<f32> = match &gpu.pos_buf {
        Some(b) => b.read_f32(|s| s.to_vec()),
        None => return,
    };
    trace_step("positions ok");

    let mut composite = vec![0.0f32; (w as usize) * (h as usize) * 4];

    // T3 Gaussian splats (back-to-front sorted).
    if let (Some(t3), Some(pb), Some(rb), Some(cb)) =
        (&gpu.t3, &gpu.pos_buf, &gpu.rad_buf, &gpu.col_buf)
    {
        use crate::frame::tiers::t3::sort_by_depth;
        let sorted = sort_by_depth(&gpu.visible, &positions, &camera);
        if !sorted.is_empty() {
            trace_step("t3.draw");
            match t3.draw(&sorted, pb, rb, cb, &camera, [w, h]) {
                Ok(pixels) => {
                    let copy_len = composite.len().min(pixels.len());
                    composite[..copy_len].copy_from_slice(&pixels[..copy_len]);
                }
                Err(e) => warn!("T3: {e}"),
            }
        }
    }

    // T2 sphere impostors (painted over T3).
    if let (Some(t2), Some(pb), Some(rb), Some(cb)) =
        (&gpu.t2, &gpu.pos_buf, &gpu.rad_buf, &gpu.col_buf)
    {
        if gpu.visible.iter().any(|(_, t)| *t == TierLevel::T2) {
            trace_step("t2.draw");
            match t2.draw(&gpu.visible, pb, rb, cb, &camera, [w, h]) {
                Ok(pixels) => {
                    // Alpha-composite T2 over T3: T2 pixel alpha in .w component.
                    for (i, chunk) in pixels.chunks(4).enumerate() {
                        if chunk.len() == 4 && chunk[3] > 0.5 {
                            let base = i * 4;
                            if base + 4 <= composite.len() {
                                composite[base..base+4].copy_from_slice(chunk);
                            }
                        }
                    }
                }
                Err(e) => warn!("T2: {e}"),
            }
        }
    }

    // §8 Edge rasterization (T1/T2/T3 visible edges).
    if let (Some(el), Some(pb), Some(csr)) =
        (&gpu.edge_line, &gpu.pos_buf, &gpu.csr)
    {
        // Build visible particle set.
        let vis_set: std::collections::HashSet<u32> =
            gpu.visible.iter().map(|&(idx, _)| idx).collect();

        // Gather edges between visible particles.
        let mut edge_list: Vec<(u32, u32)> = Vec::new();
        let flow_offs = gpu.edge.flow_offsets().to_vec();
        let mut weights: Vec<f32> = Vec::new();

        for &p in &vis_set {
            let (cols, vals) = csr.row(p as usize);
            for (&q, &w) in cols.iter().zip(vals.iter()) {
                if q > p && vis_set.contains(&q) {
                    edge_list.push((p, q));
                    weights.push(w);
                }
            }
        }

        // Resize EdgePass flow offsets if needed.
        // (flow_offs already grabbed; resize separately since we have &gpu.edge_line)
        let n_edges = edge_list.len();
        let flow_uvs: Vec<f32> = (0..n_edges)
            .map(|i| if i < flow_offs.len() { flow_offs[i] } else { 0.0 })
            .collect();

        let vp = cam.view_proj();
        trace_step("edges");
        if !edge_list.is_empty() {
            let _ = el.rasterize(
                &mut composite,
                &edge_list,
                pb,
                &weights,
                &flow_uvs,
                &vp,
                [w, h],
            );
        }
    }

    // Background: pure black for all transparent pixels.
    for chunk in composite.chunks_mut(4) {
        if chunk[3] < 0.5 {
            chunk[0] = 0.0; chunk[1] = 0.0; chunk[2] = 0.0; chunk[3] = 1.0;
        }
    }

    {
        let lit = composite.chunks(4).filter(|c| c[0] + c[1] + c[2] > 0.01).count();
        debug!("mir: composite {} lit pixels", lit);
    }
    gpu.last_pixels = Some(composite);
}


/// One-shot step tracer for bringing the pipeline up on a new driver.
fn trace_step(step: &str) {
    use std::sync::Mutex;
    static SEEN: Mutex<Vec<&'static str>> = Mutex::new(Vec::new());
    let mut seen = SEEN.lock().unwrap();
    if !seen.iter().any(|s| *s == step) {
        // leak is bounded: a handful of static step names
        seen.push(Box::leak(step.to_string().into_boxed_str()));
        debug!("mir: step {step}");
    }
}

pub fn animate_edges(mut gpu: ResMut<GpuBuffers>, time: Res<Time>) {
    let n = gpu.edge.flow_offsets().len();
    if n == 0 { return }
    let weights = vec![0.5f32; n];
    gpu.edge.update_flow_uvs(&weights, time.delta_secs());
}

pub fn composite(
    gpu:        Res<GpuBuffers>,
    mut images: ResMut<Assets<Image>>,
) {
    let (Some(pixels), Some(handle)) = (&gpu.last_pixels, &gpu.output_image) else { return };
    let Some(image) = images.get_mut(handle) else { return };
    let Some(data)  = &mut image.data else { return };

    let [w, h] = gpu.viewport;
    let expected = (w as usize) * (h as usize) * 4;
    if data.len() != expected || pixels.len() < expected { return }

    for (dst, &src) in data.iter_mut().zip(pixels.iter()) {
        *dst = (src.clamp(0.0, 1.0) * 255.0) as u8;
    }
}

// ── OnExit ────────────────────────────────────────────────────────────────────

pub fn on_exit_graph(
    mut commands: Commands,
    particles_q:  Query<Entity, With<VisibleParticle>>,
    loading_q:    Query<Entity, With<LoadingOverlay>>,
    render_q:     Query<Entity, With<RenderOutput>>,
) {
    info!("mir: exiting graph world");
    for e in particles_q.iter() { commands.entity(e).despawn(); }
    for e in loading_q.iter()   { commands.entity(e).despawn(); }
    for e in render_q.iter()    { commands.entity(e).despawn(); }
}

/// §9.4 Follow-flow: hold Alt to ride the attention current.
/// Biases camera velocity toward the strongest outgoing neighbor of the nearest particle.
pub fn follow_flow_system(
    mut cam:  ResMut<GraphCamera>,
    gpu:      Res<GpuBuffers>,
    keys:     Res<ButtonInput<KeyCode>>,
    time:     Res<Time>,
) {
    use super::camera::apply_follow_flow;
    let held = keys.pressed(KeyCode::AltLeft) || keys.pressed(KeyCode::AltRight);
    if !held { return; }
    let Some(csr) = &gpu.csr else { return };
    let Some(pb)  = &gpu.pos_buf else { return };
    if gpu.n_particles == 0 { return; }
    let positions = pb.read_f32(|s| s.to_vec());
    apply_follow_flow(&mut cam, true, &positions, csr, time.delta_secs());
}

/// §9.2 warp: consume the WarpTarget resource and initiate camera animation.
pub fn warp_to_system(
    mut cam:    ResMut<GraphCamera>,
    mut target: ResMut<WarpTarget>,
    gpu:        Res<GpuBuffers>,
) {
    use super::camera::initiate_warp;
    let Some(idx) = target.particle_idx.take() else { return };
    let Some(pb)  = &gpu.pos_buf else { return };
    let Some(rb)  = &gpu.rad_buf else { return };
    let base = idx as usize * 3;
    let center = pb.read_f32(|s| {
        if base + 2 < s.len() { [s[base], s[base+1], s[base+2]] } else { [0.0f32; 3] }
    });
    let radius = rb.read_f32(|rs| if (idx as usize) < rs.len() { rs[idx as usize] } else { 10.0 });
    let cam_pos = [center[0], center[1], center[2] + radius * 3.0];
    initiate_warp(&mut cam, cam_pos, center);
}
