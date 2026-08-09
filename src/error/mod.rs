//! # Custom warnings and errors
use std::error::Error;
use std::fmt;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Generic error to be raised when something goes wrong.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug)]
pub struct Failed {
    err: FailedError,
    msg: String,
}

/// Type of error
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Copy, Clone, Debug)]
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
    /// Error in input parameters
    ParametersError,
    /// Invalid state error (should never happen)
    InvalidStateError,
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

    /// new instance of `FailedError::ParametersError`
    pub fn input(msg: &str) -> Self {
        Failed {
            err: FailedError::ParametersError,
            msg: msg.to_string(),
        }
    }

    /// new instance of `FailedError::InvalidStateError`
    pub fn invalid_state(msg: &str) -> Self {
        Failed {
            err: FailedError::InvalidStateError,
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
            FailedError::ParametersError => "Error in input, check parameters",
            FailedError::InvalidStateError => "Invalid state, this should never happen", // useful in development phase of lib
        };
        write!(f, "{failed_err_str}")
    }
}

impl fmt::Display for Failed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.err, self.msg)
    }
}

impl Error for Failed {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;

    #[test]
    fn fit_sets_fit_failed_variant_and_message() {
        let e = Failed::fit("oops");
        assert_eq!(e.error(), FailedError::FitFailed);
        assert_eq!(e.msg, "oops");
    }

    #[test]
    fn predict_sets_predict_failed_variant_and_message() {
        let e = Failed::predict("nope");
        assert_eq!(e.error(), FailedError::PredictFailed);
        assert_eq!(e.msg, "nope");
    }

    #[test]
    fn transform_sets_transform_failed_variant_and_message() {
        let e = Failed::transform("bad");
        assert_eq!(e.error(), FailedError::TransformFailed);
        assert_eq!(e.msg, "bad");
    }

    #[test]
    fn input_sets_parameters_error_variant_and_message() {
        let e = Failed::input("no good");
        assert_eq!(e.error(), FailedError::ParametersError);
        assert_eq!(e.msg, "no good");
    }

    #[test]
    fn invalid_state_sets_invalid_state_variant_and_message() {
        let e = Failed::invalid_state("reachable?");
        assert_eq!(e.error(), FailedError::InvalidStateError);
        assert_eq!(e.msg, "reachable?");
    }

    #[test]
    fn because_sets_explicit_variant_and_message() {
        let e = Failed::because(FailedError::FindFailed, "lost");
        assert_eq!(e.error(), FailedError::FindFailed);
        assert_eq!(e.msg, "lost");
    }

    #[test]
    fn failed_error_display_each_variant() {
        assert_eq!(FailedError::FitFailed.to_string(), "Fit failed");
        assert_eq!(FailedError::PredictFailed.to_string(), "Predict failed");
        assert_eq!(FailedError::TransformFailed.to_string(), "Transform failed");
        assert_eq!(FailedError::FindFailed.to_string(), "Find failed");
        assert_eq!(
            FailedError::DecompositionFailed.to_string(),
            "Decomposition failed"
        );
        assert_eq!(
            FailedError::SolutionFailed.to_string(),
            "Can't find solution"
        );
        assert_eq!(
            FailedError::ParametersError.to_string(),
            "Error in input, check parameters"
        );
        assert_eq!(
            FailedError::InvalidStateError.to_string(),
            "Invalid state, this should never happen"
        );
    }

    #[test]
    fn failed_display_combines_variant_and_message() {
        let e = Failed::because(FailedError::FitFailed, "boom");
        assert_eq!(e.to_string(), "Fit failed: boom");
    }

    #[test]
    fn failed_error_partialeq_by_discriminant() {
        assert_eq!(FailedError::FitFailed, FailedError::FitFailed);
        assert_ne!(FailedError::FitFailed, FailedError::PredictFailed);
        // distinct variants are never equal
        let all = [
            FailedError::FitFailed,
            FailedError::PredictFailed,
            FailedError::TransformFailed,
            FailedError::FindFailed,
            FailedError::DecompositionFailed,
            FailedError::SolutionFailed,
            FailedError::ParametersError,
            FailedError::InvalidStateError,
        ];
        for (i, &a) in all.iter().enumerate() {
            for (j, &b) in all.iter().enumerate() {
                assert_eq!(a == b, i == j, "variant pair ({i}, {j}) mismatch");
            }
        }
    }

    #[test]
    fn failed_partialeq_compares_variant_and_message() {
        assert_eq!(Failed::fit("x"), Failed::fit("x"));
        assert_ne!(Failed::fit("x"), Failed::fit("y"));
        assert_ne!(Failed::fit("x"), Failed::predict("x"));
        assert_ne!(
            Failed::because(FailedError::FitFailed, "x"),
            Failed::because(FailedError::PredictFailed, "x")
        );
    }

    #[test]
    fn failed_implements_error_with_no_source() {
        let e = Failed::fit("boom");
        // Failed wraps no underlying cause
        assert!(e.source().is_none());
        // ensure it can be used as a trait object
        let _: &dyn Error = &e;
    }
}
