use serde::{Serialize, Deserialize};
use anyhow::Result;
use rayon::prelude::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataProcessor {
    pub config: ProcessorConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessorConfig {
    pub batch_size: usize,
    pub parallel_threshold: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessedData {
    pub id: u64,
    pub value: String,
    pub timestamp: u64,
}

impl DataProcessor {
    pub fn new() -> Self {
        DataProcessor {
            config: ProcessorConfig {
                batch_size: 100,
                parallel_threshold: 1000,
            },
        }
    }

    pub fn process_batch(&self, items: Vec<String>) -> Result<Vec<ProcessedData>> {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs();

        let processed: Vec<ProcessedData> = items
            .into_iter()
            .enumerate()
            .map(|(idx, value)| ProcessedData {
                id: idx as u64,
                value: value.to_uppercase(),
                timestamp,
            })
            .collect();

        Ok(processed)
    }
}

impl Default for DataProcessor {
    fn default() -> Self {
        Self::new()
    }
}

pub fn parallel_process(items: Vec<String>) -> Result<Vec<ProcessedData>> {
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_secs();

    let processed: Vec<ProcessedData> = items
        .par_iter()
        .enumerate()
        .map(|(idx, value)| ProcessedData {
            id: idx as u64,
            value: format!("PROCESSED_{}", value.to_uppercase()),
            timestamp,
        })
        .collect();

    Ok(processed)
}