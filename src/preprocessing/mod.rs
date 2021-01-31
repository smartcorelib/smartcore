/// Transform a data matrix by replaceing all categorical variables with their one-hot vector equivalents
pub mod categorical_encoders;
mod data_traits;
/// Encode a series (column, array) of categorical variables as one-hot vectors
pub mod series_encoder;
