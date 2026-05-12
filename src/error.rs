use thiserror::Error;

#[derive(Debug, Error)]
pub enum MirError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid .graph: {0}")]
    InvalidGraph(String),
    #[error("missing section `{0}`")]
    MissingSection(&'static str),
    #[error("toml: {0}")]
    Toml(#[from] toml::de::Error),
}

pub type Result<T> = std::result::Result<T, MirError>;
