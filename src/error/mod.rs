//! # Custom warnings and errors
use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};

/// Generic error to be raised when something goes wrong.
#[derive(Debug, Serialize, Deserialize)]
pub struct Failed {
    err: FailedError,
    msg: String,
}

/// Type of error
#[non_exhaustive]
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub enum FailedError {
    /// Can't fit algorithm to data
    FitFailed = 1,
    /// Can't predict new values
    PredictFailed,
    /// Can't transform data
    TransformFailed,
    /// Can't find an item
    FindFailed,
    /// Can't decompose a matrix
    DecompositionFailed,
    /// Can't solve for x
    SolutionFailed,
}

impl Failed {
    ///get type of error
    #[inline]
    pub fn error(&self) -> FailedError {
        self.err
    }

    /// new instance of `FailedError::FitError`
    pub fn fit(msg: &str) -> Self {
        Failed {
            err: FailedError::FitFailed,
            msg: msg.to_string(),
        }
    }
    /// new instance of `FailedError::PredictFailed`
    pub fn predict(msg: &str) -> Self {
        Failed {
            err: FailedError::PredictFailed,
            msg: msg.to_string(),
        }
    }

    /// new instance of `FailedError::TransformFailed`
    pub fn transform(msg: &str) -> Self {
        Failed {
            err: FailedError::TransformFailed,
            msg: msg.to_string(),
        }
    }

    /// new instance of `err`
    pub fn because(err: FailedError, msg: &str) -> Self {
        Failed {
            err,
            msg: msg.to_string(),
        }
    }
}

impl PartialEq for FailedError {
    #[inline(always)]
    fn eq(&self, rhs: &Self) -> bool {
        *self as u8 == *rhs as u8
    }
}

impl PartialEq for Failed {
    #[inline(always)]
    fn eq(&self, rhs: &Self) -> bool {
        self.err == rhs.err && self.msg == rhs.msg
    }
}

impl fmt::Display for FailedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let failed_err_str = match self {
            FailedError::FitFailed => "Fit failed",
            FailedError::PredictFailed => "Predict failed",
            FailedError::TransformFailed => "Transform failed",
            FailedError::FindFailed => "Find failed",
            FailedError::DecompositionFailed => "Decomposition failed",
            FailedError::SolutionFailed => "Can't find solution",
        };
        write!(f, "{}", failed_err_str)
    }
}

impl fmt::Display for Failed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.err, self.msg)
    }
}

impl Error for Failed {}
