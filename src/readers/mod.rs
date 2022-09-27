/// Read in from csv.
pub mod csv;

/// Error definition for readers.
mod error;
/// Utilities to help with testing functionality using IO.
/// Only meant for internal usage.
#[cfg(test)]
pub(crate) mod io_testing;

pub use error::ReadingError;
