use serde::{Deserialize, Serialize};
use reqwest;
use tokio;

#[derive(Debug)]
pub enum ApiError {
    RequestFailed(String),
    ParseError(String),
}

impl std::fmt::Display for ApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ApiError::RequestFailed(msg) => write!(f, "Request failed: {}", msg),
            ApiError::ParseError(msg) => write!(f, "Parse error: {}", msg),
        }
    }
}

impl std::error::Error for ApiError {}

#[derive(Clone)]
pub struct ApiClient {
    base_url: String,
    client: reqwest::Client,
}

impl ApiClient {
    pub fn new(base_url: String) -> Self {
        Self {
            base_url,
            client: reqwest::Client::new(),
        }
    }

    pub async fn get(&self, path: &str) -> Result<String, ApiError> {
        let url = format!("{}/{}", self.base_url, path.trim_start_matches('/'));
        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| ApiError::RequestFailed(e.to_string()))?;

        response
            .text()
            .await
            .map_err(|e| ApiError::ParseError(e.to_string()))
    }
}

pub async fn fetch_data(url: &str) -> Result<String, ApiError> {
    let client = reqwest::Client::new();
    let response = client
        .get(url)
        .send()
        .await
        .map_err(|e| ApiError::RequestFailed(e.to_string()))?;

    response
        .text()
        .await
        .map_err(|e| ApiError::ParseError(e.to_string()))
}