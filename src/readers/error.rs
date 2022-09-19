//! The module contains the errors that can happen in the `readers` folder and
//! utility functions.

/// Error wrapping all failures that can happen during loading from file.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ReadingError {
    /// The file could not be read from the file-system.
    CouldNotReadFileSystem {
        /// More details about the specific file-system error
        /// that occured.
        msg: String,
    },
    /// No rows exists in the CSV-file.
    NoRowsProvided,
    /// A field in the csv file could not be read.
    InvalidField {
        /// More details about what field could not be
        /// read and where it happened.
        msg: String,
    },
    /// A row from the csv is invalid.
    InvalidRow {
        /// More details about what row could not be read
        /// and where it happened.
        msg: String,
    },
}
impl From<std::io::Error> for ReadingError {
    fn from(io_error: std::io::Error) -> Self {
        Self::CouldNotReadFileSystem {
            msg: io_error.to_string(),
        }
    }
}
impl ReadingError {
    /// Extract the error-message from a `ReadingError`.
    pub fn message(&self) -> Option<&str> {
        match self {
            ReadingError::InvalidField { msg } => Some(msg),
            ReadingError::InvalidRow { msg } => Some(msg),
            ReadingError::CouldNotReadFileSystem { msg } => Some(msg),
            ReadingError::NoRowsProvided => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ReadingError;
    use std::io;

    #[test]
    fn reading_error_from_io_error() {
        let _parsed_reading_error: ReadingError = ReadingError::from(io::Error::new(
            io::ErrorKind::AlreadyExists,
            "File already exists .",
        ));
    }
    #[test]
    fn extract_message_from_reading_error() {
        let error_content = "Path does not exist";
        assert_eq!(
            ReadingError::CouldNotReadFileSystem {
                msg: String::from(error_content)
            }
            .message()
            .expect("This error should contain a mesage"),
            String::from(error_content)
        )
    }
}
