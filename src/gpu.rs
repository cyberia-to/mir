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
pub use wgpu_arm::{Buffer, Commands, Encoder, Gpu, GpuError, Pipeline, Queue, Shader, ShaderLib};

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
    }

    impl Gpu {
        pub fn open() -> Result<Self, GpuError> {
            let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
            let adapter = pollster::block_on(
                instance.request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    ..Default::default()
                }),
            )
            .map_err(|_| GpuError::NoAdapter)?;
            let (device, queue) = pollster::block_on(
                adapter.request_device(&wgpu::DeviceDescriptor::default()),
            )
            .map_err(|e| GpuError::Device(e.to_string()))?;
            Ok(Self { device: Arc::new(device), queue: Arc::new(queue) })
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

        pub fn buffer(&self, size: usize) -> Result<Buffer, GpuError> {
            let raw = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: size.max(4) as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            Ok(Buffer { gpu: self.clone(), raw, size })
        }

        pub fn buffer_with_data(&self, data: &[u8]) -> Result<Buffer, GpuError> {
            let raw = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: data,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });
            Ok(Buffer { gpu: self.clone(), raw, size: data.len() })
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
    }

    impl Buffer {
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
            let _ = self.gpu.device.poll(wgpu::PollType::wait_indefinitely());
            let _ = rx.recv();
            let data = slice.get_mapped_range().to_vec();
            staging.unmap();
            data
        }

        pub fn read<F, R>(&self, f: F) -> R
        where
            F: FnOnce(&[u8]) -> R,
        {
            let data = self.readback();
            f(&data[..self.size.min(data.len())])
        }

        pub fn write<F, R>(&self, f: F) -> R
        where
            F: FnOnce(&mut [u8]) -> R,
        {
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
            let data = self.readback();
            let len = (self.size.min(data.len())) / 4;
            let floats: Vec<f32> = data[..len * 4]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            f(&floats)
        }

        pub fn write_f32<F, R>(&self, f: F) -> R
        where
            F: FnOnce(&mut [f32]) -> R,
        {
            let data = self.readback();
            let len = (self.size.min(data.len())) / 4;
            let mut floats: Vec<f32> = data[..len * 4]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let r = f(&mut floats);
            let bytes: Vec<u8> = floats.iter().flat_map(|v| v.to_le_bytes()).collect();
            self.gpu.queue.write_buffer(&self.raw, 0, &bytes);
            r
        }

        pub fn as_bytes(&self) -> Vec<u8> {
            self.readback()
        }

        pub fn size(&self) -> usize {
            self.size
        }
    }
}
