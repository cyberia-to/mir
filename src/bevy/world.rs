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

/// Pixels the offscreen frame may cost. The graph is composited by compute
/// shaders and read once per frame, so its price is linear in this number —
/// it buys resolution directly out of the frame budget.
const FRAME_PIXEL_BUDGET: f32 = 640_000.0;

/// Offscreen size for a window: the window's own aspect (anything else
/// stretches the graph, since the image is drawn full-screen) at no more than
/// the budget.
fn render_size(win_w: f32, win_h: f32) -> (u32, u32) {
    let (win_w, win_h) = (win_w.max(1.0), win_h.max(1.0));
    let scale = (FRAME_PIXEL_BUDGET / (win_w * win_h)).sqrt().min(1.0);
    (
        ((win_w * scale) as u32).max(64),
        ((win_h * scale) as u32).max(64),
    )
}

pub fn on_enter_graph(
    mut commands: Commands,
    mut images:   ResMut<Assets<Image>>,
    config:       Option<Res<GraphWorldConfig>>,
    windows:      Query<&Window>,
) {
    info!("mir: entering graph world");
    let (w, h) = windows
        .single()
        .map(|win| render_size(win.width(), win.height()))
        .unwrap_or((1067, 600));
    info!("mir: frame target {w}x{h}");

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
    mut gpu:   ResMut<GpuBuffers>,
    cam:       Res<GraphCamera>,
    time:      Res<Time>,
    mut timer: Local<PassTimer>,
) {
    let dt = time.delta_secs();
    timer.frame(dt);
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

    // One frame buffer for the whole chain: splats clear and fill it, sphere
    // impostors composite over it, edges blend into it, and the CPU sees it
    // once at the end. Allocated on first use and reused every frame.
    let pixel_count = (w as usize) * (h as usize);
    if gpu.frame_buf.as_ref().map(|b| b.size()) != Some(pixel_count * 16) {
        let Some(dev) = &gpu.gpu else { return };
        match dev.buffer(pixel_count * 16) {
            Ok(b) => gpu.frame_buf = Some(b),
            Err(e) => { warn!("mir: frame buffer: {e}"); return }
        }
    }

    // T3 Gaussian splats (back-to-front sorted).
    if let (Some(t3), Some(pb), Some(rb), Some(cb), Some(fb)) =
        (&gpu.t3, &gpu.pos_buf, &gpu.rad_buf, &gpu.col_buf, &gpu.frame_buf)
    {
        use crate::frame::tiers::t3::sort_by_depth;
        let sorted = sort_by_depth(&gpu.visible, &positions, &camera);
        if !sorted.is_empty() {
            trace_step("t3.draw");
            let t0 = std::time::Instant::now();
            let drawn = t3.draw(&sorted, pb, rb, cb, &camera, [w, h], fb);
            timer.record(0, t0.elapsed().as_secs_f32() * 1000.0);
            if let Err(e) = drawn { warn!("T3: {e}"); }
        }
    }

    // T2 sphere impostors — composited over T3 inside the shader.
    if let (Some(t2), Some(pb), Some(rb), Some(cb), Some(fb)) =
        (&gpu.t2, &gpu.pos_buf, &gpu.rad_buf, &gpu.col_buf, &gpu.frame_buf)
    {
        if gpu.visible.iter().any(|(_, t)| *t == TierLevel::T2) {
            trace_step("t2.draw");
            let t0 = std::time::Instant::now();
            let drawn = t2.draw(&gpu.visible, pb, rb, cb, &camera, [w, h], fb);
            timer.record(1, t0.elapsed().as_secs_f32() * 1000.0);
            if let Err(e) = drawn { warn!("T2: {e}"); }
        }
    }

    // §8 Edge rasterization (T1/T2/T3 visible edges).
    if let (Some(el), Some(pb), Some(csr), Some(fb)) =
        (&gpu.edge_line, &gpu.pos_buf, &gpu.csr, &gpu.frame_buf)
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

        let n_edges = edge_list.len();
        let flow_uvs: Vec<f32> = (0..n_edges)
            .map(|i| if i < flow_offs.len() { flow_offs[i] } else { 0.0 })
            .collect();

        let vp = cam.view_proj();
        trace_step("edges");
        if !edge_list.is_empty() {
            let t0 = std::time::Instant::now();
            let _ = el.rasterize(fb, &edge_list, pb, &weights, &flow_uvs, &vp, [w, h]);
            timer.record(2, t0.elapsed().as_secs_f32() * 1000.0);
        }
    }

    // One sync for the whole chain, then the single readback.
    let t0 = std::time::Instant::now();
    if let (Some(dev), Some(q)) = (&gpu.gpu, &gpu.sync_queue) {
        let _ = dev.sync(q);
    }
    let Some(fb) = &gpu.frame_buf else { return };
    let mut composite = fb.read_f32(|s| s.to_vec());
    timer.record(3, t0.elapsed().as_secs_f32() * 1000.0);

    // Background: pure black for all transparent pixels.
    let t0 = std::time::Instant::now();
    for chunk in composite.chunks_mut(4) {
        if chunk[3] < 0.5 {
            chunk[0] = 0.0; chunk[1] = 0.0; chunk[2] = 0.0; chunk[3] = 1.0;
        }
    }

    timer.record(4, t0.elapsed().as_secs_f32() * 1000.0);
    timer.record(
        5,
        f32::from_bits(COMPOSITE_MS.load(std::sync::atomic::Ordering::Relaxed)),
    );

    gpu.last_pixels = Some(composite);
}


/// Per-pass frame budget, averaged over a window and logged once a second.
/// The graph world is the only place cyb can be slow, and it is slow in one
/// of four places — this says which without a profiler on the device.
#[derive(Default)]
pub struct PassTimer {
    frames: u32,
    total:  [f32; 6],
    since:  f32,
}

impl PassTimer {
    const NAMES: [&'static str; 6] =
        ["t3", "t2", "edges", "readback", "bgfill", "toimage"];

    fn record(&mut self, slot: usize, ms: f32) {
        self.total[slot] += ms;
    }

    fn frame(&mut self, dt: f32) {
        self.frames += 1;
        self.since += dt;
        if self.since < 1.0 {
            return;
        }
        let f = self.frames.max(1) as f32;
        let parts: Vec<String> = Self::NAMES
            .iter()
            .zip(self.total.iter())
            .map(|(n, t)| format!("{n} {:.1}ms", t / f))
            .collect();
        info!("mir: {:.1} fps — {}", f / self.since, parts.join(", "));
        *self = Self::default();
    }
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

    let t0 = std::time::Instant::now();
    for (dst, &src) in data.iter_mut().zip(pixels.iter()) {
        *dst = (src.clamp(0.0, 1.0) * 255.0) as u8;
    }
    COMPOSITE_MS.store(
        (t0.elapsed().as_secs_f32() * 1000.0).to_bits(),
        std::sync::atomic::Ordering::Relaxed,
    );
}

/// `composite` runs in a later schedule than `dispatch_tiers`, so it hands its
/// cost across through this cell rather than threading the timer resource
/// through two systems.
pub static COMPOSITE_MS: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(0);

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
