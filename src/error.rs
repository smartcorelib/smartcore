use std::fmt;

#[derive(Debug)]
pub struct IllegalArgumentError {
    pub message: String,
}

impl fmt::Display for IllegalArgumentError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {        
        write!(f, "{}", self.message)
    }
}