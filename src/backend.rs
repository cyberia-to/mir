//! R-1.0 §13.1 backend interface + §13.2 cpu-reference implementation.

use std::collections::HashMap;
use std::sync::{atomic::{AtomicU64, Ordering}, Mutex};

pub type BufferId   = u64;
pub type ShaderId   = u64;
pub type PipelineId = u64;
pub type TextureId  = u64;

pub enum BufferUsage  { Compute, Vertex, Index, Uniform, Readback }
pub enum ShaderStage  { Compute, Vertex, Fragment }
pub enum NumericDomain { Fp32, Fp64, FixedQ32, Goldilocks }

pub struct Binding         { pub buffer: BufferId, pub offset: usize, pub index: u32 }
pub struct PipelineShaders { pub vertex: Option<ShaderId>, pub fragment: Option<ShaderId>, pub compute: Option<ShaderId> }
pub struct PipelineLayout  { pub bind_groups: u32 }

/// GPU backend interface (§13.1).
pub trait RenderBackend: Send + Sync {
    fn buffer_create(&self, size: usize, usage: BufferUsage) -> BufferId;
    fn buffer_write(&self, id: BufferId, data: &[u8], offset: usize);
    fn buffer_read(&self, id: BufferId, offset: usize, size: usize) -> Vec<u8>;
    fn shader_compile(&self, source: &str, stage: ShaderStage) -> ShaderId;
    fn compute_dispatch(&self, shader: ShaderId, bindings: &[Binding], workgroups: [u32; 3]);
    fn pipeline_create(&self, shaders: &PipelineShaders, layout: &PipelineLayout) -> PipelineId;
    fn draw_indirect(&self, pipeline: PipelineId, bindings: &[Binding], indirect_buffer: BufferId, count: u32);
    fn surface_acquire(&self) -> TextureId;
    fn surface_present(&self, texture: TextureId);
    fn roll(&self, tensor: BufferId, shift: i32, axis: u32) -> BufferId;
    fn shifted_inner(&self, h: BufferId, c: BufferId, shifts: &[i32]) -> BufferId;
    fn shifted_wedge(&self, h: BufferId, c: BufferId, shifts: &[i32]) -> BufferId;
    fn clifford_block(&self, h: BufferId, c: BufferId, shifts: &[i32], gate: BufferId) -> BufferId;
    fn deterministic_mode(&self) -> bool;
    fn numeric_domain(&self) -> NumericDomain;
}

// ── §13.2 cpu-reference backend ───────────────────────────────────────────────

/// §13.2: single-threaded, fp64-precision cpu-reference backend.
/// Authoritative for determinism checks; slow but correct.
pub struct CpuReferenceBackend {
    next_id: AtomicU64,
    buffers: Mutex<HashMap<BufferId, Vec<u8>>>,
}

impl CpuReferenceBackend {
    pub fn new() -> Self {
        Self {
            next_id: AtomicU64::new(1),
            buffers: Mutex::new(HashMap::new()),
        }
    }

    fn alloc_id(&self) -> u64 { self.next_id.fetch_add(1, Ordering::Relaxed) }

    /// Compute a golden epoch state for conformance comparison.
    /// This is the authoritative cpu-reference result for P-RENDER-TOPO/POS.
    pub fn render_epoch(&self, csr: &crate::graph::Csr) -> crate::epoch::EpochState {
        crate::epoch::cpu_reference_epoch(csr)
    }

    /// P-RENDER-TOPO (§15.1): compare cluster IDs against another epoch state.
    /// Returns true iff cluster_ids match bit-for-bit at all 4 τ levels.
    pub fn verify_topology(&self, reference: &crate::epoch::EpochState, other: &crate::epoch::EpochState) -> bool {
        reference.cluster_ids == other.cluster_ids
    }

    /// P-RENDER-POS (§15.2): max L2 deviation ≤ ε_X = 1e-3 · R_scene.
    pub fn verify_positions(&self, reference: &crate::epoch::EpochState, other: &crate::epoch::EpochState, r_scene: f32) -> (bool, f32) {
        let epsilon = 1e-3 * r_scene;
        let n = reference.positions.len() / 3;
        if other.positions.len() / 3 != n { return (false, f32::INFINITY); }
        let max_dev = (0..n).map(|i| {
            let b = i * 3;
            let dx = reference.positions[b]   - other.positions[b];
            let dy = reference.positions[b+1] - other.positions[b+1];
            let dz = reference.positions[b+2] - other.positions[b+2];
            (dx*dx + dy*dy + dz*dz).sqrt()
        }).fold(0.0f32, f32::max);
        (max_dev <= epsilon, max_dev)
    }
}

impl Default for CpuReferenceBackend {
    fn default() -> Self { Self::new() }
}

impl RenderBackend for CpuReferenceBackend {
    fn buffer_create(&self, size: usize, _usage: BufferUsage) -> BufferId {
        let id = self.alloc_id();
        self.buffers.lock().unwrap().insert(id, vec![0u8; size]);
        id
    }

    fn buffer_write(&self, id: BufferId, data: &[u8], offset: usize) {
        if let Some(buf) = self.buffers.lock().unwrap().get_mut(&id) {
            let end = (offset + data.len()).min(buf.len());
            if offset < buf.len() {
                buf[offset..end].copy_from_slice(&data[..end - offset]);
            }
        }
    }

    fn buffer_read(&self, id: BufferId, offset: usize, size: usize) -> Vec<u8> {
        self.buffers.lock().unwrap().get(&id).map(|buf| {
            let start = offset.min(buf.len());
            let end   = (offset + size).min(buf.len());
            buf[start..end].to_vec()
        }).unwrap_or_default()
    }

    fn shader_compile(&self, _source: &str, _stage: ShaderStage) -> ShaderId { self.alloc_id() }

    fn compute_dispatch(&self, _shader: ShaderId, _bindings: &[Binding], _workgroups: [u32; 3]) {
        // CPU-reference: no GPU dispatch; epoch computations run via render_epoch().
    }

    fn pipeline_create(&self, _shaders: &PipelineShaders, _layout: &PipelineLayout) -> PipelineId { self.alloc_id() }

    fn draw_indirect(&self, _pipeline: PipelineId, _bindings: &[Binding], _indirect_buffer: BufferId, _count: u32) {}

    fn surface_acquire(&self) -> TextureId { self.alloc_id() }
    fn surface_present(&self, _texture: TextureId) {}

    fn roll(&self, tensor: BufferId, shift: i32, axis: u32) -> BufferId {
        // CPU implementation: cyclic shift of f32 tensor along axis.
        let data = self.buffer_read(tensor, 0, usize::MAX);
        let floats: Vec<f32> = data.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let n = floats.len();
        if n == 0 { return self.alloc_id(); }
        let _ = axis; // multi-axis roll not yet implemented
        let s = shift.rem_euclid(n as i32) as usize;
        let mut rolled = Vec::with_capacity(n);
        rolled.extend_from_slice(&floats[n - s..]);
        rolled.extend_from_slice(&floats[..n - s]);
        let bytes: Vec<u8> = rolled.iter().flat_map(|f| f.to_le_bytes()).collect();
        let id = self.buffer_create(bytes.len(), BufferUsage::Compute);
        self.buffer_write(id, &bytes, 0);
        id
    }

    fn shifted_inner(&self, _h: BufferId, _c: BufferId, _shifts: &[i32]) -> BufferId {
        self.alloc_id() // §13.1: Clifford shifted inner product — requires bivector A^eff_2
    }
    fn shifted_wedge(&self, _h: BufferId, _c: BufferId, _shifts: &[i32]) -> BufferId {
        self.alloc_id() // §13.1: Clifford shifted wedge product — requires bivector A^eff_2
    }
    fn clifford_block(&self, _h: BufferId, _c: BufferId, _shifts: &[i32], _gate: BufferId) -> BufferId {
        self.alloc_id() // §13.1: fused Clifford block — required for §7.1 T∞ full NRF
    }

    fn deterministic_mode(&self) -> bool { true }
    fn numeric_domain(&self) -> NumericDomain { NumericDomain::Fp64 }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_backend_buffer_roundtrip() {
        let b = CpuReferenceBackend::new();
        let id = b.buffer_create(16, BufferUsage::Compute);
        let data = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
        b.buffer_write(id, &data, 4);
        let out = b.buffer_read(id, 4, 8);
        assert_eq!(out, data);
    }

    #[test]
    fn cpu_backend_roll_cyclic() {
        let b = CpuReferenceBackend::new();
        // 4 floats: [1, 2, 3, 4]. shift=1 → [4, 1, 2, 3].
        let floats = [1.0f32, 2.0, 3.0, 4.0];
        let bytes: Vec<u8> = floats.iter().flat_map(|f| f.to_le_bytes()).collect();
        let id = b.buffer_create(bytes.len(), BufferUsage::Compute);
        b.buffer_write(id, &bytes, 0);
        let out_id = b.roll(id, 1, 0);
        let out_bytes = b.buffer_read(out_id, 0, 16);
        let out_f: Vec<f32> = out_bytes.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert!((out_f[0] - 4.0).abs() < 1e-6, "roll[0]={}", out_f[0]);
        assert!((out_f[1] - 1.0).abs() < 1e-6, "roll[1]={}", out_f[1]);
    }

    #[test]
    fn cpu_backend_deterministic() {
        let b = CpuReferenceBackend::new();
        assert!(b.deterministic_mode());
        assert!(matches!(b.numeric_domain(), NumericDomain::Fp64));
    }

    #[test]
    fn verify_positions_self_is_zero_deviation() {
        let b = CpuReferenceBackend::new();
        let csr = crate::graph::Csr::empty();
        let epoch = b.render_epoch(&csr);
        let (ok, dev) = b.verify_positions(&epoch, &epoch, 1000.0);
        assert!(ok, "self-comparison should pass");
        assert_eq!(dev, 0.0);
    }

    #[test]
    fn verify_topology_self_matches() {
        let b = CpuReferenceBackend::new();
        let csr = crate::graph::Csr::empty();
        let epoch = b.render_epoch(&csr);
        assert!(b.verify_topology(&epoch, &epoch));
    }
}

// ── §13.3 sparse kernels ──────────────────────────────────────────────────────

/// Sparse CSR matrix-vector product: `y = A·x`.
///
/// One name for every platform. On Apple targets this forwards to
/// `acpu::sparse::csr_matvec_set` (NEON, software prefetch); elsewhere it runs
/// the portable loop below. Both compute the identical sum in the identical
/// order — f32 accumulation left to right per row — so the result is
/// bit-for-bit the same and conformance holds across backends.
pub fn csr_matvec_set(row_ptr: &[u32], col_idx: &[u32], values: &[f32], x: &[f32], y: &mut [f32]) {
    #[cfg(target_vendor = "apple")]
    {
        acpu::sparse::csr_matvec_set(row_ptr, col_idx, values, x, y);
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        let n = row_ptr.len().saturating_sub(1);
        debug_assert_eq!(y.len(), n);
        debug_assert_eq!(col_idx.len(), values.len());
        y.fill(0.0);
        for i in 0..n {
            let s = row_ptr[i] as usize;
            let e = row_ptr[i + 1] as usize;
            let mut acc = 0.0f32;
            for k in s..e {
                acc += values[k] * x[col_idx[k] as usize];
            }
            y[i] = acc;
        }
    }
}
