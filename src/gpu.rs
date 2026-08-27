//! Device facade — one name for the GPU layer, per platform.
//!
//! On Apple targets this re-exports [`aruminium`] unchanged: the same types,
//! the same zero-copy Metal path, no indirection. Everywhere else the same
//! surface is implemented over wgpu (Vulkan on Android/Linux, DX12 on
//! Windows): correctness-first — readback goes through a staging copy, the
//! zero-copy unimem import is portable-backends.md P4. Pass sources select
//! MSL or WGSL per arm via [`ShaderSource`]-style cfg alongside each kernel;
//! feeding MSL to this arm simply fails `compile`, which the pass
//! constructors already treat as "this pass stays off".

#[cfg(target_vendor = "apple")]
pub use aruminium::{Buffer, Commands, Encoder, Gpu, GpuError, Pipeline, Queue, Shader, ShaderLib};

#[cfg(not(target_vendor = "apple"))]
pub use wgpu_arm::{
    Buffer, Commands, Encoder, FrameReader, Gpu, GpuError, Pipeline, Queue, Shader, ShaderLib,
    install_shared,
};

/// Per-frame readback of the packed frame, unified-memory arm: one sync,
/// one mapped copy — the readback that was always cheap here stays sync.
#[cfg(target_vendor = "apple")]
#[derive(Default)]
pub struct FrameReader;

#[cfg(target_vendor = "apple")]
impl FrameReader {
    pub fn new() -> Self { Self }

    pub fn fetch(&mut self, gpu: &Gpu, queue: &Queue, buf: &Buffer, dst: &mut Vec<u8>) -> bool {
        let _ = gpu.sync(queue);
        buf.read(|b| {
            let n = b.len().min(dst.len());
            dst[..n].copy_from_slice(&b[..n]);
        });
        true
    }
}

#[cfg(not(target_vendor = "apple"))]
mod wgpu_arm {
    use std::fmt;
    use std::sync::{Arc, Mutex};

    use wgpu::util::DeviceExt;

    #[derive(Debug, Clone)]
    pub enum GpuError {
        NoAdapter,
        Device(String),
        Compile(String),
        Function(String),
        Internal(String),
    }

    impl fmt::Display for GpuError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                GpuError::NoAdapter => write!(f, "no wgpu adapter available"),
                GpuError::Device(e) => write!(f, "wgpu device: {e}"),
                GpuError::Compile(e) => write!(f, "shader compile: {e}"),
                GpuError::Function(e) => write!(f, "shader function: {e}"),
                GpuError::Internal(e) => write!(f, "gpu internal: {e}"),
            }
        }
    }

    impl std::error::Error for GpuError {}

    #[derive(Clone)]
    pub struct Gpu {
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        /// MAPPABLE_PRIMARY_BUFFERS is live on this device: storage buffers
        /// carry MAP_READ|MAP_WRITE and the closures below touch the
        /// allocation directly — no staging buffer, no GPU copy. Bevy's own
        /// device already has it on integrated GPUs (its Functionality
        /// priority takes every adapter feature and only strips this one on
        /// discrete cards, where the PCI-E round-trip would hurt).
        mappable: bool,
    }

    fn probe(device: &wgpu::Device) -> bool {
        let mappable = device.features().contains(wgpu::Features::MAPPABLE_PRIMARY_BUFFERS);
        log::info!("gpu: direct-map {}", if mappable { "on" } else { "off (staging copies)" });
        mappable
    }

    /// Prove the unimem→GPU seam on this device, once, out loud: write a
    /// pattern into a pinned block, wrap it, read it back through the GPU.
    /// On the zero-copy path the GPU is reading the block's own pages; on
    /// the fallback it reads the copy — either way the bytes must match.
    /// One-shot inventory of the external-memory extensions this driver
    /// offers versus what the device was actually created with — the two
    /// facts that decide whether a zero-copy import is reachable at all.
    #[cfg(any(target_os = "android", target_os = "linux"))]
    fn report_external_memory(gpu: &Gpu) {
        unsafe {
            let Some(hal) = gpu.device.as_hal::<wgpu::hal::api::Vulkan>() else { return };
            let enabled: Vec<&str> = hal
                .enabled_device_extensions()
                .iter()
                .filter_map(|n| n.to_str().ok())
                .filter(|n| n.contains("external_memory") || n.contains("external_fence"))
                .collect();
            let supported: Vec<String> = hal
                .shared_instance()
                .raw_instance()
                .enumerate_device_extension_properties(hal.raw_physical_device())
                .map(|props| {
                    props
                        .iter()
                        .filter_map(|p| p.extension_name_as_c_str().ok()?.to_str().ok().map(String::from))
                        .filter(|n| n.contains("external_memory"))
                        .collect()
                })
                .unwrap_or_default();
            log::info!("gpu: external-memory supported {supported:?}");
            log::info!("gpu: external-memory enabled {enabled:?}");
        }
    }

    #[cfg(not(any(target_os = "android", target_os = "linux")))]
    fn report_external_memory(_gpu: &Gpu) {}

    fn verify_unimem_wrap(gpu: &Gpu) {
        report_external_memory(gpu);
        let Ok(block) = unimem::Block::open(64 * 1024) else {
            log::warn!("gpu: unimem probe — block allocation failed");
            return;
        };
        for (i, w) in block.as_f32_mut().iter_mut().enumerate() {
            *w = i as f32;
        }
        // The import path is the whole point of the probe: say plainly which
        // one engaged, and when it did not, why — that reason is the only
        // thing separating a zero-copy device from a copying one.
        let how = match gpu.import_host(&block) {
            Ok(_) => "zero-copy import".to_string(),
            Err(why) => format!("copy fallback: {why}"),
        };
        match gpu.wrap(&block) {
            Ok(buffer) => {
                let ok = buffer.read_f32(|s| {
                    s.len() >= 3 && s[0] == 0.0 && s[1] == 1.0 && s[2] == 2.0
                });
                log::info!(
                    "gpu: unimem wrap {} ({how})",
                    if ok { "verified" } else { "MISMATCH" },
                );
            }
            Err(e) => log::warn!("gpu: unimem probe — wrap failed: {e}"),
        }
    }

    static GLOBAL: std::sync::OnceLock<Result<Gpu, GpuError>> = std::sync::OnceLock::new();

    /// Hand the facade an existing device (e.g. Bevy's render device) before
    /// the first `Gpu::open()`. One `VkDevice` per process is not just about
    /// wgpu ids: a second device sharing the GPU has produced driver-level
    /// hangs on PowerVR (Pixel 10) where MoltenVK tolerated it.
    pub fn install_shared(device: wgpu::Device, queue: wgpu::Queue) -> bool {
        let mappable = probe(&device);
        let gpu = Gpu { device: Arc::new(device), queue: Arc::new(queue), mappable };
        verify_unimem_wrap(&gpu);
        GLOBAL.set(Ok(gpu)).is_ok()
    }

    impl Gpu {
        /// One device per process, like Metal's system default device: every
        /// pass calls `open()` and they must all land on the same `Device`,
        /// or buffers from one pass cannot bind into another's pipeline
        /// (wgpu-core panics on the cross-hub id).
        pub fn open() -> Result<Self, GpuError> {
            GLOBAL
                .get_or_init(|| {
                    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
                    let adapter = pollster::block_on(
                        instance.request_adapter(&wgpu::RequestAdapterOptions {
                            power_preference: wgpu::PowerPreference::HighPerformance,
                            ..Default::default()
                        }),
                    )
                    .map_err(|_| GpuError::NoAdapter)?;
                    let features = adapter.features()
                        & wgpu::Features::MAPPABLE_PRIMARY_BUFFERS;
                    let (device, queue) = pollster::block_on(
                        adapter.request_device(&wgpu::DeviceDescriptor {
                            required_features: features,
                            ..Default::default()
                        }),
                    )
                    .map_err(|e| GpuError::Device(e.to_string()))?;
                    let mappable = probe(&device);
                    Ok(Gpu { device: Arc::new(device), queue: Arc::new(queue), mappable })
                })
                .clone()
        }

        /// Compile WGSL. An MSL source (a pass not yet ported) fails
        /// validation here and the pass stays off — by design.
        pub fn compile(&self, source: &str) -> Result<ShaderLib, GpuError> {
            self.device.push_error_scope(wgpu::ErrorFilter::Validation);
            let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
            if let Some(e) = pollster::block_on(self.device.pop_error_scope()) {
                return Err(GpuError::Compile(e.to_string()));
            }
            Ok(ShaderLib { gpu: self.clone(), module })
        }

        pub fn pipeline(&self, function: &Shader) -> Result<Pipeline, GpuError> {
            self.device.push_error_scope(wgpu::ErrorFilter::Validation);
            let raw = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&function.entry),
                layout: None,
                module: &function.module,
                entry_point: Some(&function.entry),
                compilation_options: Default::default(),
                cache: None,
            });
            if let Some(e) = pollster::block_on(self.device.pop_error_scope()) {
                return Err(GpuError::Compile(e.to_string()));
            }
            Ok(Pipeline { raw })
        }

        pub fn new_command_queue(&self) -> Result<Queue, GpuError> {
            Ok(Queue { gpu: self.clone() })
        }

        /// Block until everything submitted has finished. Submissions on one
        /// queue execute in order, so passes can record and submit without
        /// stalling individually — only the reader waits, once.
        pub fn sync(&self, _queue: &Queue) -> Result<(), GpuError> {
            let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
            Ok(())
        }

        fn storage_usage(&self) -> wgpu::BufferUsages {
            let mut usage = wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST;
            if self.mappable {
                usage |= wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::MAP_WRITE;
            }
            usage
        }

        pub fn buffer(&self, size: usize) -> Result<Buffer, GpuError> {
            let raw = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: size.max(4) as u64,
                usage: self.storage_usage(),
                mapped_at_creation: false,
            });
            Ok(Buffer { gpu: self.clone(), raw, size, import: None })
        }

        pub fn buffer_with_data(&self, data: &[u8]) -> Result<Buffer, GpuError> {
            let raw = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: data,
                usage: self.storage_usage(),
            });
            Ok(Buffer { gpu: self.clone(), raw, size: data.len(), import: None })
        }
    }

    pub struct ShaderLib {
        gpu: Gpu,
        module: wgpu::ShaderModule,
    }

    impl ShaderLib {
        pub fn function(&self, name: &str) -> Result<Shader, GpuError> {
            let _ = &self.gpu;
            Ok(Shader { module: self.module.clone(), entry: name.to_string() })
        }
    }

    pub struct Shader {
        module: wgpu::ShaderModule,
        entry: String,
    }

    pub struct Pipeline {
        raw: wgpu::ComputePipeline,
    }

    pub struct Queue {
        gpu: Gpu,
    }

    impl Queue {
        pub fn commands(&self) -> Result<Commands, GpuError> {
            let encoder = self
                .gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            Ok(Commands {
                gpu: self.gpu.clone(),
                encoder: Arc::new(Mutex::new(Some(encoder))),
            })
        }
    }

    pub struct Commands {
        gpu: Gpu,
        encoder: Arc<Mutex<Option<wgpu::CommandEncoder>>>,
    }

    impl Commands {
        pub fn encoder(&self) -> Result<Encoder, GpuError> {
            Ok(Encoder {
                gpu: self.gpu.clone(),
                encoder: self.encoder.clone(),
                state: Mutex::new(EncState { pipeline: None, entries: Vec::new() }),
            })
        }

        pub fn submit(&self) {
            if let Some(encoder) = self.encoder.lock().unwrap().take() {
                self.gpu.queue.submit([encoder.finish()]);
            }
        }

        pub fn wait(&self) {
            let _ = self.gpu.device.poll(wgpu::PollType::wait_indefinitely());
        }
    }

    struct EncState {
        pipeline: Option<wgpu::ComputePipeline>,
        /// (binding index, buffer) recorded by `bind_buffer`/`push`.
        entries: Vec<(u32, wgpu::Buffer)>,
    }

    pub struct Encoder {
        gpu: Gpu,
        encoder: Arc<Mutex<Option<wgpu::CommandEncoder>>>,
        state: Mutex<EncState>,
    }

    impl Encoder {
        pub fn bind(&self, pipeline: &Pipeline) {
            self.state.lock().unwrap().pipeline = Some(pipeline.raw.clone());
        }

        pub fn bind_buffer(&self, buffer: &Buffer, offset: usize, index: usize) {
            debug_assert_eq!(offset, 0, "facade carries whole-buffer bindings");
            self.state.lock().unwrap().entries.push((index as u32, buffer.raw.clone()));
        }

        /// Metal's `setBytes`: small inline constants. Here: a uniform buffer
        /// at the same binding index — the WGSL side declares `var<uniform>`.
        pub fn push(&self, data: &[u8], index: usize) {
            // Uniform binding sizes round up to 16.
            let mut padded = data.to_vec();
            while padded.len() % 16 != 0 {
                padded.push(0);
            }
            let raw = self.gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: &padded,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
            self.state.lock().unwrap().entries.push((index as u32, raw));
        }

        /// Metal's `dispatchThreads`: an exact grid. WebGPU dispatches whole
        /// workgroups, so this rounds up — every WGSL kernel carries its own
        /// bounds guard.
        pub fn launch(&self, grid: (usize, usize, usize), group: (usize, usize, usize)) {
            let groups = (
                grid.0.div_ceil(group.0.max(1)) as u32,
                grid.1.div_ceil(group.1.max(1)) as u32,
                grid.2.div_ceil(group.2.max(1)) as u32,
            );
            self.dispatch(groups);
        }

        pub fn launch_groups(&self, groups: (usize, usize, usize), _threads: (usize, usize, usize)) {
            self.dispatch((groups.0 as u32, groups.1 as u32, groups.2 as u32));
        }

        fn dispatch(&self, groups: (u32, u32, u32)) {
            let state = self.state.lock().unwrap();
            let Some(pipeline) = &state.pipeline else { return };

            let layout = pipeline.get_bind_group_layout(0);
            let entries: Vec<wgpu::BindGroupEntry> = state
                .entries
                .iter()
                .map(|(binding, buffer)| wgpu::BindGroupEntry {
                    binding: *binding,
                    resource: buffer.as_entire_binding(),
                })
                .collect();
            let bind_group = self.gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &layout,
                entries: &entries,
            });

            let mut guard = self.encoder.lock().unwrap();
            let Some(encoder) = guard.as_mut() else { return };
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(groups.0, groups.1, groups.2);
        }

        pub fn finish(&self) {}
    }

    pub struct Buffer {
        gpu: Gpu,
        raw: wgpu::Buffer,
        size: usize,
        /// Set when the storage is an imported unimem block: the CPU view
        /// belongs to the block's owner, wgpu cannot map it, and the raw
        /// Vulkan handles below outlive the wgpu buffer and are freed on
        /// drop after a device wait.
        import: Option<HostImport>,
    }

    impl Buffer {
        /// Wait for the map callback: try_recv + poll in a loop. A single
        /// poll(Wait) can return before the callback registers (seen on the
        /// Pixel 10's PowerVR), and nothing else is guaranteed to poll this
        /// device — the render thread may be parked in the pipelined-
        /// rendering rendezvous.
        fn pump_map(&self, rx: &std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>) {
            loop {
                match rx.try_recv() {
                    Ok(_) => break,
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => break,
                    Err(std::sync::mpsc::TryRecvError::Empty) => {
                        let _ = self.gpu.device.poll(wgpu::PollType::wait_indefinitely());
                    }
                }
            }
        }

        /// Map the buffer itself and run `f` over its bytes — the direct
        /// path, zero staging. Caller contract (same as Metal shared
        /// storage): no GPU work in flight on this buffer.
        fn with_mapped_read<F, R>(&self, f: F) -> R
        where
            F: FnOnce(&[u8]) -> R,
        {
            let slice = self.raw.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            self.pump_map(&rx);
            let r = f(&slice.get_mapped_range());
            self.raw.unmap();
            r
        }

        fn with_mapped_write<F, R>(&self, f: F) -> R
        where
            F: FnOnce(&mut [u8]) -> R,
        {
            let slice = self.raw.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Write, move |r| {
                let _ = tx.send(r);
            });
            self.pump_map(&rx);
            let r = {
                let mut view = slice.get_mapped_range_mut();
                f(&mut view)
            };
            self.raw.unmap();
            r
        }

        /// The staging fallback for devices without MAPPABLE_PRIMARY_BUFFERS.
        fn readback(&self) -> Vec<u8> {
            let staging = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: self.raw.size(),
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let mut encoder = self
                .gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            encoder.copy_buffer_to_buffer(&self.raw, 0, &staging, 0, self.raw.size());
            self.gpu.queue.submit([encoder.finish()]);

            let slice = staging.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            self.pump_map(&rx);
            let data = slice.get_mapped_range().to_vec();
            staging.unmap();
            data
        }

        pub fn read<F, R>(&self, f: F) -> R
        where
            F: FnOnce(&[u8]) -> R,
        {
            if self.gpu.mappable && self.import.is_none() {
                return self.with_mapped_read(|bytes| f(&bytes[..self.size.min(bytes.len())]));
            }
            let data = self.readback();
            f(&data[..self.size.min(data.len())])
        }

        pub fn write<F, R>(&self, f: F) -> R
        where
            F: FnOnce(&mut [u8]) -> R,
        {
            if self.gpu.mappable && self.import.is_none() {
                return self.with_mapped_write(|bytes| {
                    let len = self.size.min(bytes.len());
                    f(&mut bytes[..len])
                });
            }
            let mut data = self.readback();
            let len = self.size.min(data.len());
            let r = f(&mut data[..len]);
            self.gpu.queue.write_buffer(&self.raw, 0, &data);
            r
        }

        pub fn read_f32<F, R>(&self, f: F) -> R
        where
            F: FnOnce(&[f32]) -> R,
        {
            self.read(|bytes| {
                let len = bytes.len() / 4;
                debug_assert_eq!(bytes.as_ptr() as usize % 4, 0, "mapped range under-aligned");
                let floats =
                    unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, len) };
                f(floats)
            })
        }

        pub fn write_f32<F, R>(&self, f: F) -> R
        where
            F: FnOnce(&mut [f32]) -> R,
        {
            self.write(|bytes| {
                let len = bytes.len() / 4;
                debug_assert_eq!(bytes.as_ptr() as usize % 4, 0, "mapped range under-aligned");
                let floats =
                    unsafe { std::slice::from_raw_parts_mut(bytes.as_mut_ptr() as *mut f32, len) };
                f(floats)
            })
        }

        pub fn as_bytes(&self) -> Vec<u8> {
            self.read(|b| b.to_vec())
        }

        pub fn size(&self) -> usize {
            self.size
        }
    }

    /// Latency-for-throughput frame readback: copy into a persistent staging
    /// buffer and map it asynchronously — this frame consumes the *previous*
    /// frame's pixels and never blocks on the GPU. The synchronous path here
    /// drained the whole device queue (bevy's frame included) every frame,
    /// which alone was two thirds of the frame budget on the Pixel 10.
    pub struct FrameReader {
        staging: Option<wgpu::Buffer>,
        pending: Option<std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>,
    }

    impl FrameReader {
        pub fn new() -> Self {
            Self { staging: None, pending: None }
        }

        /// Copy the newest completed frame into `dst`. Returns true when a
        /// fresh frame landed; false leaves last frame's pixels standing.
        pub fn fetch(&mut self, _gpu: &Gpu, _q: &Queue, buf: &Buffer, dst: &mut Vec<u8>) -> bool {
            let device = &buf.gpu.device;
            let size = buf.raw.size();
            if self.staging.as_ref().map(|s| s.size()) != Some(size) {
                self.staging = Some(device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("frame-staging"),
                    size,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
                self.pending = None;
            }
            let staging = self.staging.as_ref().unwrap();

            let mut fresh = false;
            if let Some(rx) = &self.pending {
                let _ = device.poll(wgpu::PollType::Poll);
                match rx.try_recv() {
                    Ok(Ok(())) => {
                        let slice = staging.slice(..);
                        let view = slice.get_mapped_range();
                        let n = view.len().min(dst.len());
                        dst[..n].copy_from_slice(&view[..n]);
                        drop(view);
                        staging.unmap();
                        self.pending = None;
                        fresh = true;
                    }
                    Ok(Err(_)) | Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                        self.pending = None;
                    }
                    Err(std::sync::mpsc::TryRecvError::Empty) => {}
                }
            }

            // Queue order does the synchronization: the copy lands after every
            // pass already submitted this frame, and the map callback after
            // the copy. No device-wide wait anywhere.
            if self.pending.is_none() {
                let mut encoder = device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                encoder.copy_buffer_to_buffer(&buf.raw, 0, staging, 0, size);
                buf.gpu.queue.submit([encoder.finish()]);
                let (tx, rx) = std::sync::mpsc::channel();
                staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                    let _ = tx.send(r);
                });
                let _ = device.poll(wgpu::PollType::Poll);
                self.pending = Some(rx);
            }
            fresh
        }
    }

    /// The imported `VkDeviceMemory` behind a wrapped unimem block.
    ///
    /// Ownership is split: `Buffer::from_raw` leaves the memory to the caller
    /// but wgpu still destroys the `VkBuffer` itself on drop, so this frees
    /// the memory only — destroying the buffer here too is a double-free the
    /// PowerVR driver answers with a SIGSEGV inside `vkDestroyBuffer`.
    struct HostImport {
        device: Arc<wgpu::Device>,
        memory: ash::vk::DeviceMemory,
    }

    unsafe impl Send for HostImport {}
    unsafe impl Sync for HostImport {}

    impl Drop for HostImport {
        fn drop(&mut self) {
            // The wgpu buffer is dropped first (field order in Buffer puts
            // `raw` before `import`); wait out in-flight GPU work before
            // freeing the memory it was bound to.
            let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
            unsafe {
                if let Some(hal) = self.device.as_hal::<wgpu::hal::api::Vulkan>() {
                    hal.raw_device().free_memory(self.memory, None);
                }
            }
        }
    }

    impl Gpu {
        /// Wrap a pinned unimem block as a GPU buffer.
        ///
        /// The zero-copy path imports the block's pages into Vulkan via
        /// `VK_EXT_external_memory_host` — the CPU view stays the block's
        /// mmap, the GPU reads the same physical pages, and nothing is
        /// copied. That needs the extension enabled at device creation
        /// (cyb registers a bevy raw-vulkan callback for it); anywhere the
        /// chain is missing this falls back to one copy of the block's
        /// current contents, which is exactly what `buffer_with_data` does.
        pub fn wrap(&self, block: &unimem::Block) -> Result<Buffer, GpuError> {
            match self.import_host(block) {
                Ok(buffer) => {
                    log::debug!("gpu: wrap zero-copy ({} bytes)", block.size());
                    Ok(buffer)
                }
                Err(why) => {
                    log::debug!("gpu: wrap copies ({why})");
                    self.buffer_with_data(block.as_bytes())
                }
            }
        }

        #[cfg(not(any(target_os = "android", target_os = "linux")))]
        fn import_host(&self, _block: &unimem::Block) -> Result<Buffer, String> {
            Err("host import is vulkan-only".into())
        }

        /// Zero-copy import, best available handle type first.
        ///
        /// Android's AHardwareBuffer is the platform IOSurface — gralloc
        /// memory the CPU has locked and the GPU can bind. Desktop/embedded
        /// Vulkan instead offers host-pointer import of an ordinary mmap.
        /// A driver may have neither (PowerVR on the Pixel 10 has no
        /// host-pointer import), which is what the copy fallback is for.
        #[cfg(any(target_os = "android", target_os = "linux"))]
        fn import_host(&self, block: &unimem::Block) -> Result<Buffer, String> {
            #[cfg(target_os = "android")]
            match self.import_ahardware(block) {
                Ok(buffer) => return Ok(buffer),
                Err(why) => {
                    log::debug!("gpu: AHardwareBuffer import unavailable ({why})");
                }
            }
            self.import_host_pointer(block)
        }

        #[cfg(any(target_os = "android", target_os = "linux"))]
        fn import_host_pointer(&self, block: &unimem::Block) -> Result<Buffer, String> {
            use ash::vk;

            unsafe {
                let hal = self
                    .device
                    .as_hal::<wgpu::hal::api::Vulkan>()
                    .ok_or("backend is not vulkan")?;
                if !hal
                    .enabled_device_extensions()
                    .contains(&ash::ext::external_memory_host::NAME)
                {
                    // Separate "the driver cannot" from "we failed to ask":
                    // the first is the end of the road on this device, the
                    // second is a bug in the create-device callback.
                    let supported = hal
                        .shared_instance()
                        .raw_instance()
                        .enumerate_device_extension_properties(hal.raw_physical_device())
                        .map(|props| {
                            props.iter().any(|p| {
                                p.extension_name_as_c_str()
                                    .is_ok_and(|n| n == ash::ext::external_memory_host::NAME)
                            })
                        })
                        .unwrap_or(false);
                    return Err(if supported {
                        "VK_EXT_external_memory_host supported but not enabled at device creation"
                            .into()
                    } else {
                        "VK_EXT_external_memory_host unsupported by this driver".to_string()
                    });
                }
                let dev = hal.raw_device();
                let instance = hal.shared_instance().raw_instance();
                let phys = hal.raw_physical_device();

                // Host-pointer import wants minImportedHostPointerAlignment;
                // the block's mmap is page-aligned, which satisfies the
                // universal 4 KB and the 16 KB kernels alike.
                let mut host_props = vk::PhysicalDeviceExternalMemoryHostPropertiesEXT::default();
                let mut props2 =
                    vk::PhysicalDeviceProperties2::default().push_next(&mut host_props);
                instance.get_physical_device_properties2(phys, &mut props2);
                let align = host_props.min_imported_host_pointer_alignment as usize;
                let ptr = block.address() as *mut std::ffi::c_void;
                if align == 0 || (ptr as usize) % align != 0 {
                    return Err(format!("block not aligned to {align}"));
                }
                let import_size = block.alloc_size();
                if import_size % align != 0 {
                    return Err(format!("block allocation not a multiple of {align}"));
                }

                let ext_fns = ash::ext::external_memory_host::Device::new(instance, dev);
                let mut ptr_props = vk::MemoryHostPointerPropertiesEXT::default();
                (ext_fns.fp().get_memory_host_pointer_properties_ext)(
                    dev.handle(),
                    vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT,
                    ptr,
                    &mut ptr_props,
                )
                .result()
                .map_err(|e| format!("host pointer properties: {e}"))?;

                let mut ext_buf = vk::ExternalMemoryBufferCreateInfo::default()
                    .handle_types(vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT);
                let buf_info = vk::BufferCreateInfo::default()
                    .size(import_size as u64)
                    .usage(
                        vk::BufferUsageFlags::STORAGE_BUFFER
                            | vk::BufferUsageFlags::TRANSFER_SRC
                            | vk::BufferUsageFlags::TRANSFER_DST,
                    )
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .push_next(&mut ext_buf);
                let raw_buf = dev
                    .create_buffer(&buf_info, None)
                    .map_err(|e| format!("create_buffer: {e}"))?;

                let req = dev.get_buffer_memory_requirements(raw_buf);
                let mem_props = instance.get_physical_device_memory_properties(phys);
                let type_bits = req.memory_type_bits & ptr_props.memory_type_bits;
                let wanted =
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
                let index = (0..mem_props.memory_type_count)
                    .filter(|i| type_bits & (1 << i) != 0)
                    .find(|i| {
                        mem_props.memory_types[*i as usize]
                            .property_flags
                            .contains(wanted)
                    })
                    .or_else(|| (0..mem_props.memory_type_count).find(|i| type_bits & (1 << i) != 0));
                let Some(index) = index else {
                    dev.destroy_buffer(raw_buf, None);
                    return Err("no importable memory type".into());
                };

                let mut import_info = vk::ImportMemoryHostPointerInfoEXT::default()
                    .handle_type(vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT)
                    .host_pointer(ptr);
                let alloc_info = vk::MemoryAllocateInfo::default()
                    .allocation_size(import_size as u64)
                    .memory_type_index(index)
                    .push_next(&mut import_info);
                let memory = match dev.allocate_memory(&alloc_info, None) {
                    Ok(m) => m,
                    Err(e) => {
                        dev.destroy_buffer(raw_buf, None);
                        return Err(format!("allocate_memory(import): {e}"));
                    }
                };
                if let Err(e) = dev.bind_buffer_memory(raw_buf, memory, 0) {
                    dev.destroy_buffer(raw_buf, None);
                    dev.free_memory(memory, None);
                    return Err(format!("bind_buffer_memory: {e}"));
                }

                let hal_buffer = wgpu::hal::vulkan::Buffer::from_raw(raw_buf);
                let buffer = self.device.create_buffer_from_hal::<wgpu::hal::api::Vulkan>(
                    hal_buffer,
                    &wgpu::BufferDescriptor {
                        label: Some("unimem-import"),
                        size: import_size as u64,
                        usage: wgpu::BufferUsages::STORAGE
                            | wgpu::BufferUsages::COPY_SRC
                            | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    },
                );

                Ok(Buffer {
                    gpu: self.clone(),
                    raw: buffer,
                    size: block.size(),
                    import: Some(HostImport { device: self.device.clone(), memory }),
                })
            }
        }

        #[cfg(target_os = "android")]
        fn import_ahardware(&self, block: &unimem::Block) -> Result<Buffer, String> {
            use ash::vk;

            unsafe {
                let hal = self
                    .device
                    .as_hal::<wgpu::hal::api::Vulkan>()
                    .ok_or("backend is not vulkan")?;
                let ext_name = ash::android::external_memory_android_hardware_buffer::NAME;
                if !hal.enabled_device_extensions().contains(&ext_name) {
                    return Err(format!("{} not enabled on device", ext_name.to_string_lossy()));
                }
                let dev = hal.raw_device();
                let instance = hal.shared_instance().raw_instance();
                let ahb = block.handle() as *mut vk::AHardwareBuffer;

                let ext_fns =
                    ash::android::external_memory_android_hardware_buffer::Device::new(instance, dev);
                let mut props = vk::AndroidHardwareBufferPropertiesANDROID::default();
                (ext_fns.fp().get_android_hardware_buffer_properties_android)(
                    dev.handle(),
                    ahb,
                    &mut props,
                )
                .result()
                .map_err(|e| format!("AHardwareBuffer properties: {e}"))?;

                let mut ext_buf = vk::ExternalMemoryBufferCreateInfo::default()
                    .handle_types(vk::ExternalMemoryHandleTypeFlags::ANDROID_HARDWARE_BUFFER_ANDROID);
                let buf_info = vk::BufferCreateInfo::default()
                    .size(block.alloc_size() as u64)
                    .usage(
                        vk::BufferUsageFlags::STORAGE_BUFFER
                            | vk::BufferUsageFlags::TRANSFER_SRC
                            | vk::BufferUsageFlags::TRANSFER_DST,
                    )
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .push_next(&mut ext_buf);
                let raw_buf = dev
                    .create_buffer(&buf_info, None)
                    .map_err(|e| format!("create_buffer: {e}"))?;

                let req = dev.get_buffer_memory_requirements(raw_buf);
                let type_bits = req.memory_type_bits & props.memory_type_bits;
                let Some(index) = (0..32).find(|i| type_bits & (1 << i) != 0) else {
                    dev.destroy_buffer(raw_buf, None);
                    return Err("no importable memory type".into());
                };

                let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().buffer(raw_buf);
                let mut import_info =
                    vk::ImportAndroidHardwareBufferInfoANDROID::default().buffer(ahb);
                let alloc_info = vk::MemoryAllocateInfo::default()
                    .allocation_size(props.allocation_size)
                    .memory_type_index(index)
                    .push_next(&mut import_info)
                    .push_next(&mut dedicated);
                let memory = match dev.allocate_memory(&alloc_info, None) {
                    Ok(m) => m,
                    Err(e) => {
                        dev.destroy_buffer(raw_buf, None);
                        return Err(format!("allocate_memory(import AHB): {e}"));
                    }
                };
                if let Err(e) = dev.bind_buffer_memory(raw_buf, memory, 0) {
                    dev.destroy_buffer(raw_buf, None);
                    dev.free_memory(memory, None);
                    return Err(format!("bind_buffer_memory: {e}"));
                }

                let hal_buffer = wgpu::hal::vulkan::Buffer::from_raw(raw_buf);
                let buffer = self.device.create_buffer_from_hal::<wgpu::hal::api::Vulkan>(
                    hal_buffer,
                    &wgpu::BufferDescriptor {
                        label: Some("unimem-ahb"),
                        size: block.alloc_size() as u64,
                        usage: wgpu::BufferUsages::STORAGE
                            | wgpu::BufferUsages::COPY_SRC
                            | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    },
                );

                Ok(Buffer {
                    gpu: self.clone(),
                    raw: buffer,
                    size: block.size(),
                    import: Some(HostImport { device: self.device.clone(), memory }),
                })
            }
        }
    }
}
