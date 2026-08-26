//! Device facade — one name for the GPU layer, per platform.
//!
//! On Apple targets this re-exports [`aruminium`] unchanged: the same types,
//! the same zero-copy Metal path, no indirection. Everywhere else it provides
//! the same surface with every constructor returning [`GpuError::Unavailable`],
//! so the pass constructors' existing `Result` handling degrades exactly like
//! a runtime `Gpu::open()` failure does on a Mac without Metal — the world
//! runs, the GPU paint does not. The portable wgpu implementation lands here
//! (portable-backends.md P3).

#[cfg(target_vendor = "apple")]
pub use aruminium::{Buffer, Commands, Encoder, Gpu, GpuError, Pipeline, Queue, Shader, ShaderLib};

#[cfg(not(target_vendor = "apple"))]
pub use stub::{Buffer, Commands, Encoder, Gpu, GpuError, Pipeline, Queue, Shader, ShaderLib};

/// The no-device stand-in. None of these values can be constructed — every
/// entry point returns [`GpuError::Unavailable`] — so the method bodies are
/// statically unreachable; they exist to keep the call sites type-checking.
#[cfg(not(target_vendor = "apple"))]
#[allow(dead_code)] // surface exists to type-check call sites; nothing constructs these
mod stub {
    use std::fmt;

    /// Uninhabited: makes non-constructibility a type-level fact.
    #[derive(Debug, Clone, Copy)]
    enum Void {}

    #[derive(Debug, Clone)]
    pub enum GpuError {
        /// No device layer exists on this platform yet.
        Unavailable,
    }

    impl fmt::Display for GpuError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "gpu unavailable on this platform (mir portable backend pending)")
        }
    }

    impl std::error::Error for GpuError {}

    macro_rules! void_type {
        ($name:ident) => {
            #[derive(Debug)]
            pub struct $name {
                void: Void,
            }
            impl $name {
                fn absurd(&self) -> ! {
                    match self.void {}
                }
            }
        };
    }

    void_type!(Gpu);
    void_type!(Buffer);
    void_type!(Pipeline);
    void_type!(Queue);
    void_type!(Commands);
    void_type!(Encoder);
    void_type!(ShaderLib);
    void_type!(Shader);

    unsafe impl Send for Gpu {}
    unsafe impl Sync for Gpu {}
    unsafe impl Send for Buffer {}
    unsafe impl Sync for Buffer {}
    unsafe impl Send for Pipeline {}
    unsafe impl Sync for Pipeline {}
    unsafe impl Send for Queue {}
    unsafe impl Sync for Queue {}

    impl Gpu {
        pub fn open() -> Result<Self, GpuError> {
            Err(GpuError::Unavailable)
        }
        pub fn compile(&self, _source: &str) -> Result<ShaderLib, GpuError> {
            self.absurd()
        }
        pub fn pipeline(&self, _function: &Shader) -> Result<Pipeline, GpuError> {
            self.absurd()
        }
        pub fn new_command_queue(&self) -> Result<Queue, GpuError> {
            self.absurd()
        }
        pub fn buffer(&self, _size: usize) -> Result<Buffer, GpuError> {
            self.absurd()
        }
        pub fn buffer_with_data(&self, _data: &[u8]) -> Result<Buffer, GpuError> {
            self.absurd()
        }
    }

    impl ShaderLib {
        pub fn function(&self, _name: &str) -> Result<Shader, GpuError> {
            self.absurd()
        }
    }

    impl Queue {
        pub fn commands(&self) -> Result<Commands, GpuError> {
            self.absurd()
        }
    }

    impl Commands {
        pub fn encoder(&self) -> Result<Encoder, GpuError> {
            self.absurd()
        }
        pub fn submit(&self) {
            self.absurd()
        }
        pub fn wait(&self) {
            self.absurd()
        }
    }

    impl Encoder {
        pub fn bind(&self, _pipeline: &Pipeline) {
            self.absurd()
        }
        pub fn bind_buffer(&self, _buffer: &Buffer, _offset: usize, _index: usize) {
            self.absurd()
        }
        pub fn push(&self, _data: &[u8], _index: usize) {
            self.absurd()
        }
        pub fn launch(&self, _grid: (usize, usize, usize), _group: (usize, usize, usize)) {
            self.absurd()
        }
        pub fn launch_groups(&self, _groups: (usize, usize, usize), _threads: (usize, usize, usize)) {
            self.absurd()
        }
        pub fn finish(&self) {
            self.absurd()
        }
    }

    impl Buffer {
        pub fn read<F, R>(&self, _f: F) -> R
        where
            F: FnOnce(&[u8]) -> R,
        {
            self.absurd()
        }
        pub fn write<F, R>(&self, _f: F) -> R
        where
            F: FnOnce(&mut [u8]) -> R,
        {
            self.absurd()
        }
        pub fn read_f32<F, R>(&self, _f: F) -> R
        where
            F: FnOnce(&[f32]) -> R,
        {
            self.absurd()
        }
        pub fn write_f32<F, R>(&self, _f: F) -> R
        where
            F: FnOnce(&mut [f32]) -> R,
        {
            self.absurd()
        }
        pub fn as_bytes(&self) -> &[u8] {
            self.absurd()
        }
        pub fn size(&self) -> usize {
            self.absurd()
        }
    }
}
