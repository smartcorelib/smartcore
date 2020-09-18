//! # Custom warnings and errors
use std::error::Error;
use std::fmt;

/// Error to be raised when model does not fits data.
#[derive(Debug)]
pub struct FitFailedError {
    details: String,
}

/// Error to be raised when model prediction cannot be calculated.
#[derive(Debug)]
pub struct PredictFailedError {
    details: String,
}

impl FitFailedError {
    /// Creates new instance of `FitFailedError`
    /// * `msg` - description of the error
    pub fn new(msg: &str) -> FitFailedError {
        FitFailedError {
            details: msg.to_string(),
        }
    }
}

impl fmt::Display for FitFailedError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.details)
    }
}

impl Error for FitFailedError {
    fn description(&self) -> &str {
        &self.details
    }
}

impl PredictFailedError {
    /// Creates new instance of `PredictFailedError`
    /// * `msg` - description of the error
    pub fn new(msg: &str) -> PredictFailedError {
        PredictFailedError {
            details: msg.to_string(),
        }
    }
}

impl fmt::Display for PredictFailedError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.details)
    }
}

impl Error for PredictFailedError {
    fn description(&self) -> &str {
        &self.details
    }
}
