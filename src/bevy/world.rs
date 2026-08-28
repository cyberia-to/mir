//! Graph world systems.

use std::sync::{Arc, RwLock};
use bevy::asset::RenderAssetUsages;
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages};

use crate::epoch::EpochWorker;
use crate::frame::diffusion::diffusion_step;

use super::resources::{EpochStateRes, GpuBuffers, GraphCamera, GraphWorldConfig, WarpTarget};

#[derive(States, Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum GraphWorldState { #[default] Inactive, Active }

#[derive(Component)] pub struct LoadingOverlay;
#[derive(Component)] pub struct RenderOutput;

// ── OnEnter ─────────────────────────────────────────────────────────────────

/// Pixels the offscreen frame may cost. The graph is composited by compute
/// shaders and read once per frame, so its price is linear in this number —
/// it buys resolution directly out of the frame budget.
///
/// High enough that a phone and a laptop both paint one texel per screen
/// pixel; it is a safety valve for very large displays, not a normal limit.
/// Anything lower shows up as blur, and worse: tier is assigned by screen
/// diameter, so a frame painted at half resolution demotes every node one
/// tier and the graph loses its spheres before it loses its sharpness.
const FRAME_PIXEL_BUDGET: f32 = 4_200_000.0;

/// Offscreen size for a window: the window's own aspect (anything else
/// stretches the graph, since the image is drawn full-screen) at no more than
/// the budget.
///
/// Takes *physical* pixels. Logical ones are the same number on a 1x monitor
/// and a fraction of it everywhere else — 443x986 on a 1080x2404 phone, which
/// is the whole image stretched 2.4x before it reaches the eye.
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
    // Whatever the window says right now is a guess. On Android the real
    // surface size arrives a frame or two after the world opens — enter early
    // enough and the window still reports the size the app *asked* for, which
    // on a phone is a landscape desktop window that fits no screen at all.
    // `track_frame_size` corrects this as soon as the truth shows up; here we
    // only need something valid to allocate.
    let (w, h) = windows
        .single()
        .map(|win| render_size(win.resolution.physical_width() as f32,
                               win.resolution.physical_height() as f32))
        .unwrap_or((1067, 600));
    info!("mir: frame target {w}x{h}");

    // Create blank RGBA8 output image.
    let mut image = Image::new(
        Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        TextureDimension::D2,
        // Pure black until the first composite lands — a grey clear
        // shows through as a grey background on the first frames.
        vec![0u8; (w * h * 4) as usize],
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

/// Cull, depth-sort and edge-gather — only when the camera actually moved.
/// A still camera re-uses last frame's caches whole: no cull dispatch, no
/// sort, no edge set rebuild, and (on the readback arms) no GPU stalls.
pub fn sync_visible_entities(mut gpu: ResMut<GpuBuffers>, cam: Res<GraphCamera>) {
    if gpu.n_particles == 0 { return }
    let camera = cam.to_gpu_camera();
    if gpu.cached_vp == Some(camera.view_proj) { return }

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

    gpu.sorted = crate::frame::paint::sort_by_depth(&visible, &gpu.pos_cpu, &camera);

    // Edges between visible particles, undirected, deduped by (min,max).
    let vis_set: std::collections::HashSet<u32> =
        visible.iter().map(|&(idx, _)| idx).collect();
    let (mut edge_list, mut weights) = (Vec::new(), Vec::new());
    if let Some(csr) = &gpu.csr {
        for &p in &vis_set {
            let (cols, vals) = csr.row(p as usize);
            for (&q, &w) in cols.iter().zip(vals.iter()) {
                if q > p && vis_set.contains(&q) {
                    edge_list.push((p, q));
                    weights.push(w);
                }
            }
        }
    }
    debug!("mir: cull -> {} visible, {} edges", visible.len(), edge_list.len());
    gpu.segments = crate::frame::paint::edge_segments(
        &edge_list, &weights, &gpu.pos_cpu, &camera, gpu.viewport);
    gpu.edge_list = edge_list;
    gpu.edge_weights = weights;
    gpu.visible = visible;
    gpu.cached_vp = Some(camera.view_proj);
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
    let pixel_count = (w as usize) * (h as usize);

    // The frame is one packed RGBA8 buffer, written by the single paint
    // dispatch and mapped (asynchronously) by the reader — nothing else.
    if gpu.frame_u8.as_ref().map(|b| b.size()) != Some(pixel_count * 4) {
        let Some(dev) = &gpu.gpu else { return };
        match dev.buffer(pixel_count * 4) {
            Ok(b8) => gpu.frame_u8 = Some(b8),
            Err(e) => { warn!("mir: frame buffer: {e}"); return }
        }
    }

    // One kernel, one command buffer, one submission per frame.
    let Some(cmd) = gpu.sync_queue.as_ref().and_then(|q| q.commands().ok()) else { return };
    if let (Some(paint), Some(f8)) = (&gpu.paint, &gpu.frame_u8) {
        trace_step("paint");
        let t0 = std::time::Instant::now();
        let drawn = paint.draw(&gpu.sorted, &gpu.visible,
                               &gpu.pos_cpu, &gpu.rad_cpu, &gpu.col_cpu,
                               &gpu.segments, &camera, [w, h], f8, &cmd);
        timer.record(0, t0.elapsed().as_secs_f32() * 1000.0);
        if let Err(e) = drawn { warn!("paint: {e}"); }
    }
    cmd.submit();
    let t0 = std::time::Instant::now();
    let mut pixels = gpu.last_pixels.take().unwrap_or_default();
    pixels.resize(pixel_count * 4, 0);
    {
        let g = &mut *gpu;
        if let (Some(dev), Some(q), Some(f8)) = (&g.gpu, &g.sync_queue, &g.frame_u8) {
            g.reader.fetch(dev, q, f8, &mut pixels);
        }
    }
    timer.record(1, t0.elapsed().as_secs_f32() * 1000.0);
    timer.record(
        2,
        f32::from_bits(COMPOSITE_MS.load(std::sync::atomic::Ordering::Relaxed)),
    );

    gpu.last_pixels = Some(pixels);
}


/// Per-pass frame budget, averaged over a window and logged once a second.
/// The graph world is the only place cyb can be slow, and it is slow in one
/// of four places — this says which without a profiler on the device.
#[derive(Default)]
pub struct PassTimer {
    frames: u32,
    total:  [f32; 3],
    since:  f32,
}

impl PassTimer {
    const NAMES: [&'static str; 3] = ["paint", "readback", "toimage"];

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

/// Keep the offscreen frame the size of the window it is drawn into.
///
/// Two things make this necessary rather than nice. A window that is resized
/// keeps painting at its old size and gets stretched. And on Android the size
/// at world-entry is not the surface size yet, so the first frame target is
/// simply wrong — large enough, in the case that prompted this, to fail wgpu
/// validation and take the app down.
///
/// The GPU-side buffer already resizes itself off `viewport`; the image has to
/// be told, and `composite` skips any frame where the two disagree, so a size
/// change costs one dropped frame and nothing else.
pub fn track_frame_size(
    mut gpu:    ResMut<GpuBuffers>,
    mut images: ResMut<Assets<Image>>,
    windows:    Query<&Window>,
) {
    let Ok(win) = windows.single() else { return };
    let (w, h) = render_size(win.resolution.physical_width() as f32,
                             win.resolution.physical_height() as f32);
    if gpu.viewport == [w, h] { return }

    let Some(handle) = gpu.output_image.clone() else { return };
    let Some(image) = images.get_mut(&handle) else { return };
    image.texture_descriptor.size = Extent3d {
        width: w, height: h, depth_or_array_layers: 1,
    };
    image.data = Some(vec![0u8; (w as usize) * (h as usize) * 4]);
    gpu.viewport = [w, h];
    info!("mir: frame target {w}x{h}");
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
    data.copy_from_slice(&pixels[..expected]);
    dump_frame_once(&pixels[..expected], w, h);
    COMPOSITE_MS.store(
        (t0.elapsed().as_secs_f32() * 1000.0).to_bits(),
        std::sync::atomic::Ordering::Relaxed,
    );
}

/// Write one settled frame to `$MIR_DUMP_FRAME` as a binary PPM, then never
/// again. Unset, this costs an atomic load per frame and does nothing.
///
/// The graph is the one part of cyb whose bugs are only visible as pixels, and
/// the honest way to look at those pixels is to look at the ones the renderer
/// produced — not at a photograph of a screen, which adds a compositor, a
/// scale factor and a colour profile between the defect and the eye. PPM
/// because it is eight lines of code and every tool reads it.
fn dump_frame_once(rgba: &[u8], w: u32, h: u32) {
    use std::sync::atomic::{AtomicU32, Ordering};
    static FRAME: AtomicU32 = AtomicU32::new(0);
    // Let the epoch settle and the camera stop moving first; frame zero is
    // black and frame one is half a graph.
    const SETTLED: u32 = 150;

    let n = FRAME.fetch_add(1, Ordering::Relaxed);
    if n != SETTLED { return }
    let Ok(path) = std::env::var("MIR_DUMP_FRAME") else { return };

    let mut out = format!("P6\n{w} {h}\n255\n").into_bytes();
    out.reserve(rgba.len() / 4 * 3);
    for px in rgba.chunks_exact(4) {
        out.extend_from_slice(&px[..3]);
    }
    match std::fs::write(&path, out) {
        Ok(())  => info!("mir: frame dumped to {path} ({w}x{h})"),
        Err(e)  => warn!("mir: frame dump to {path} failed: {e}"),
    }
}

/// `composite` runs in a later schedule than `dispatch_tiers`, so it hands its
/// cost across through this cell rather than threading the timer resource
/// through two systems.
pub static COMPOSITE_MS: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(0);

// ── OnExit ────────────────────────────────────────────────────────────────────

pub fn on_exit_graph(
    mut commands: Commands,
    loading_q:    Query<Entity, With<LoadingOverlay>>,
    render_q:     Query<Entity, With<RenderOutput>>,
) {
    info!("mir: exiting graph world");
    for e in loading_q.iter() { commands.entity(e).despawn(); }
    for e in render_q.iter()  { commands.entity(e).despawn(); }
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
    if gpu.n_particles == 0 { return; }
    apply_follow_flow(&mut cam, true, &gpu.pos_cpu, csr, time.delta_secs());
}

/// §9.2 warp: consume the WarpTarget resource and initiate camera animation.
pub fn warp_to_system(
    mut cam:    ResMut<GraphCamera>,
    mut target: ResMut<WarpTarget>,
    gpu:        Res<GpuBuffers>,
) {
    use super::camera::initiate_warp;
    let Some(idx) = target.particle_idx.take() else { return };
    let base = idx as usize * 3;
    let center: [f32; 3] = match gpu.pos_cpu.get(base..base + 3) {
        Some(s) => [s[0], s[1], s[2]],
        None => return,
    };
    let radius = gpu.rad_cpu.get(idx as usize).copied().unwrap_or(10.0);
    let cam_pos = [center[0], center[1], center[2] + radius * 3.0];
    initiate_warp(&mut cam, cam_pos, center);
}
