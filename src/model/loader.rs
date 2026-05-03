use crate::layers::TransformerLayer;
use crate::model::config::ModelConfig;
use crate::model::{KvCache, Model};
use crate::quant::gguf::parse_header;
use crate::quant::dequantize;
use crate::tensor::Tensor;
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;

#[derive(Debug)]
pub enum LoaderError {
    Io(std::io::Error),
    InvalidFormat(String),
    UnsupportedArchitecture(String),
    QuantizationNotImplemented(String),
    MissingTensor(String),
}

impl From<std::io::Error> for LoaderError {
    fn from(e: std::io::Error) -> Self {
        LoaderError::Io(e)
    }
}

impl From<String> for LoaderError {
    fn from(e: String) -> Self {
        LoaderError::InvalidFormat(e)
    }
}

/// Extract a String value from GGUF metadata.
fn get_str<'a>(pairs: &'a [crate::quant::gguf::GgufKv], key: &str) -> Option<&'a str> {
    pairs.iter().find(|kv| kv.key == key).and_then(|kv| {
        if let crate::quant::gguf::GgufValue::String(s) = &kv.value {
            Some(s.as_str())
        } else {
            None
        }
    })
}

fn get_u32(pairs: &[crate::quant::gguf::GgufKv], key: &str) -> Option<u32> {
    pairs.iter().find(|kv| kv.key == key).and_then(|kv| match &kv.value {
        crate::quant::gguf::GgufValue::U32(v) => Some(*v),
        crate::quant::gguf::GgufValue::I32(v) => Some(*v as u32),
        crate::quant::gguf::GgufValue::U64(v) => Some(*v as u32),
        _ => None,
    })
}

fn get_f32(pairs: &[crate::quant::gguf::GgufKv], key: &str) -> Option<f32> {
    pairs.iter().find(|kv| kv.key == key).and_then(|kv| match &kv.value {
        crate::quant::gguf::GgufValue::F32(v) => Some(*v),
        crate::quant::gguf::GgufValue::F64(v) => Some(*v as f32),
        _ => None,
    })
}

fn get_u64(pairs: &[crate::quant::gguf::GgufKv], key: &str) -> Option<u64> {
    pairs.iter().find(|kv| kv.key == key).and_then(|kv| match &kv.value {
        crate::quant::gguf::GgufValue::U64(v) => Some(*v),
        crate::quant::gguf::GgufValue::I64(v) => Some(*v as u64),
        crate::quant::gguf::GgufValue::U32(v) => Some(*v as u64),
        _ => None,
    })
}

/// Extract the architecture string (llama, gemma, etc.) from GGUF metadata.
fn detect_arch(pairs: &[crate::quant::gguf::GgufKv]) -> Result<String, LoaderError> {
    // Try both naming conventions
    if let Some(arch) = get_str(pairs, "general.architecture") {
        return Ok(arch.to_lowercase());
    }
    // Fallback: check if it's a known architecture via other keys
    for kv in pairs {
        if kv.key.starts_with("llama.") {
            return Ok("llama".into());
        }
    }
    Err(LoaderError::InvalidFormat("unknown architecture".into()))
}

/// Load a model from a GGUF file using mmap.
pub fn load_gguf(path: &str) -> Result<Model, LoaderError> {
    let file = File::open(path)?;
    let mmap = unsafe { Mmap::map(&file)? };
    let data = &mmap[..];

    let header = parse_header(data)?;

    // Build config from KV metadata
    let arch = detect_arch(&header.kv_pairs)?;
    let mut config = ModelConfig::guess_defaults(&arch);

    // Override with actual values from the file
    if let Some(v) = get_u32(&header.kv_pairs, &format!("{arch}.block_count")) {
        config.n_layers = v as usize;
    }
    if let Some(v) = get_u32(&header.kv_pairs, &format!("{arch}.embedding_length")) {
        config.n_embd = v as usize;
    }
    if let Some(v) = get_u32(&header.kv_pairs, &format!("{arch}.feed_forward_length")) {
        config.n_intermediate = v as usize;
    }
    if let Some(v) = get_u32(&header.kv_pairs, &format!("{arch}.attention.head_count")) {
        config.n_heads = v as usize;
    }
    if let Some(v) = get_u32(&header.kv_pairs, &format!("{arch}.attention.head_count_kv")) {
        config.n_kv_heads = v as usize;
    } else {
        config.n_kv_heads = config.n_heads; // default MHA
    }
    if let Some(v) = get_u32(&header.kv_pairs, &format!("{arch}.context_length")) {
        config.max_seq_len = v as usize;
    }
    if let Some(v) = get_u32(&header.kv_pairs, &format!("{arch}.vocab_size")) {
        config.vocab_size = v as usize;
    }
    if let Some(v) = get_f32(&header.kv_pairs, &format!("{arch}.attention.layer_norm_rms_epsilon")) {
        config.norm_eps = v;
    }
    if let Some(v) = get_f32(&header.kv_pairs, &format!("{arch}.rope.freq_base")) {
        config.rope_theta = v;
    }

    config.head_dim = config.n_embd / config.n_heads;
    config.n_gqa = config.n_heads / config.n_kv_heads;

    // Build tensor name → info map
    let tensor_map: HashMap<&str, &crate::quant::gguf::GgufTensorInfo> = header
        .tensor_infos
        .iter()
        .map(|t| (t.name.as_str(), t))
        .collect();

    // Helper: read and dequantize a tensor by name
    let load_tensor = |name: &str| -> Result<Tensor, LoaderError> {
        let info = tensor_map.get(name).ok_or_else(|| LoaderError::MissingTensor(name.to_string()))?;
        let start = header.data_offset as usize + info.offset as usize;
        let numel: usize = info.shape.iter().map(|&d| d as usize).product();
        let bs = info.quant_type.block_size();
        let bp = info.quant_type.bytes_per_block();
        let num_blocks = (numel + bs - 1) / bs;
        let end = start + num_blocks * bp;
        let raw = &data[start..end.min(data.len())];
        let values = dequantize(raw, info.quant_type, numel);
        // GGUF stores shape innermost-first (GGML convention); our Tensor is row-major outermost-first.
        let shape: Vec<usize> = info.shape.iter().rev().map(|&d| d as usize).collect();
        Ok(Tensor::from_vec(values, shape))
    };

    // Load embeddings
    let tok_embeddings = load_tensor("token_embd.weight")?;
    let output_weight = load_tensor("output.weight")?;
    let norm = load_tensor("output_norm.weight")?;

    // Load layers
    let mut layers = Vec::with_capacity(config.n_layers);
    let mut kv_caches = Vec::with_capacity(config.n_layers);

    for i in 0..config.n_layers {
        let prefix = format!("blk.{i}");
        let mut layer = TransformerLayer::new(&config);

        // Fill weights from loaded tensors
        layer.attention_norm = load_tensor(&format!("{prefix}.attn_norm.weight"))?;
        layer.ffn_norm = load_tensor(&format!("{prefix}.ffn_norm.weight"))?;

        // Attention weights
        layer.attention.wq = load_tensor(&format!("{prefix}.attn_q.weight"))?;
        layer.attention.wk = load_tensor(&format!("{prefix}.attn_k.weight"))?;
        layer.attention.wv = load_tensor(&format!("{prefix}.attn_v.weight"))?;
        layer.attention.wo = load_tensor(&format!("{prefix}.attn_output.weight"))?;

        // FFN weights
        layer.ffn.gate = load_tensor(&format!("{prefix}.ffn_gate.weight"))?;
        layer.ffn.up = load_tensor(&format!("{prefix}.ffn_up.weight"))?;
        layer.ffn.down = load_tensor(&format!("{prefix}.ffn_down.weight"))?;

        layer.norm_eps = config.norm_eps;

        layers.push(layer);
        kv_caches.push(KvCache::new(
            config.max_seq_len,
            config.n_kv_heads,
            config.head_dim,
        ));
    }

    let norm_eps = config.norm_eps;
    Ok(Model {
        config,
        layers,
        tok_embeddings,
        output_weight,
        norm,
        norm_eps,
        kv_caches,
    })
}
