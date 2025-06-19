
use std::fmt;

#[derive(Debug)]
pub enum GpuError {
    NoAdapter(String),
    NoShader,
    NoDevice(String),
    InvalidWorkgroupSize,
    MutexLock(String),
    WorkerConversion,
    ParamsBufferNotFound,
    Generic(String)
}

impl std::error::Error for GpuError {}
impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoAdapter(err) => write!(f, "Unable to create GPU adapter, error: {}", err),
            Self::NoShader => write!(f, "GPU adapter does not support computer shaders."),
            Self::NoDevice(err) => write!(f, "Unable to create device on GPU, error: {}", err),
            Self::InvalidWorkgroupSize => write!(f, "Workgroup size must be 64, 128, 256, 512 or 1024"),
            Self::MutexLock(msg) => write!(f, "Unable to lock mutex: {}", msg),
            Self::WorkerConversion => write!(f, "Unable to convert into GpuWorker"), 
            Self::ParamsBufferNotFound => write!(f, "Unable to update params buffer, as there doesn't appear to be a params buffer in this worker!"),
            Self::Generic(msg) => write!(f, "{}", msg),
        }
    }
}

impl From<std::io::Error> for GpuError {
    fn from(err: std::io::Error) -> Self {
        GpuError::Generic(err.to_string())
    }
}

