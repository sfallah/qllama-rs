use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplitData {
    pub split_id: usize,
    pub no_tokens: usize,
    pub split_embedding_name: String,
    pub split_embedding_file: String,
    pub split_string: String,
    pub sentence_embeddings_name: String,
    pub sentence_embeddings_file: String,
    pub sentences: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SummaryData {
    pub text: String,
    pub summary: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuerySummaries {
    pub query: String,
    pub summaries: Vec<String>,
}
