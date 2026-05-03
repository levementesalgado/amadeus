use serde::{Deserialize, Serialize};

/// Hyper-parameters of a transformer model, as found in a GGUF/config file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub name: String,

    // Architecture
    pub architecture: String,         // e.g. "llama", "gemma", "phi3"
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub n_embd: usize,                // hidden / d_model
    pub n_intermediate: usize,        // FFN hidden dimension
    pub head_dim: usize,              // n_embd / n_heads
    pub max_seq_len: usize,
    pub vocab_size: usize,
    pub norm_eps: f32,                // e.g. 1e-5

    // Quantization
    pub quant_type: String,           // "Q4_0", "Q8_0", "F16", "F32", etc.
    pub n_gqa: usize,                 // grouped-query attention groups

    // Misc
    pub rope_theta: f32,              // RoPE base frequency
    pub rope_scaling: Option<RopeScaling>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RopeScaling {
    pub rope_type: String,            // "linear", "dynamic", "yarn"
    pub factor: f32,
}

impl ModelConfig {
    /// Guess from architecture name if not explicitly set.
    pub fn guess_defaults(arch: &str) -> Self {
        // Reasonable defaults for common architectures
        match arch {
            "llama" => Self {
                name: arch.into(),
                architecture: arch.into(),
                n_layers: 32,
                n_heads: 32,
                n_kv_heads: 8,
                n_embd: 4096,
                n_intermediate: 11008,
                head_dim: 128,
                max_seq_len: 4096,
                vocab_size: 32000,
                norm_eps: 1e-5,
                quant_type: "Q4_0".into(),
                n_gqa: 4,
                rope_theta: 500000.0,
                rope_scaling: None,
            },
            "gemma" => Self {
                name: arch.into(),
                architecture: arch.into(),
                n_layers: 18,
                n_heads: 8,
                n_kv_heads: 1,
                n_embd: 2048,
                n_intermediate: 16384,
                head_dim: 256,
                max_seq_len: 8192,
                vocab_size: 256000,
                norm_eps: 1e-6,
                quant_type: "Q4_0".into(),
                n_gqa: 1,
                rope_theta: 10000.0,
                rope_scaling: None,
            },
            _ => Self {
                name: arch.into(),
                architecture: arch.into(),
                n_layers: 16,
                n_heads: 16,
                n_kv_heads: 16,
                n_embd: 2048,
                n_intermediate: 8192,
                head_dim: 128,
                max_seq_len: 2048,
                vocab_size: 32000,
                norm_eps: 1e-5,
                quant_type: "F16".into(),
                n_gqa: 1,
                rope_theta: 10000.0,
                rope_scaling: None,
            },
        }
    }
}
