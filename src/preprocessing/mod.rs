/// Transform a data matrix by replacing all categorical variables with their one-hot vector equivalents
pub mod categorical;
mod traits;
/// Preprocess numerical matrices.
pub mod numerical;
/// Encode a series (column, array) of categorical variables as one-hot vectors
pub mod series_encoder;
