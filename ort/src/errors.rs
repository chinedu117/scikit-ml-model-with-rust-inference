use std::fmt;


#[derive(Debug, Clone)]
pub enum AppError {
    Message(String),
    Session(String),
}

impl std::error::Error for AppError {}

impl fmt::Display for AppError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.clone() {
            AppError::Message(e) => write!(f, "{}", e),
            AppError::Session(e) => write!(f, "{}", e),
        }
    }
}

