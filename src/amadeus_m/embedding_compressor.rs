use std::collections::HashMap;
use std::fs::File;
use memmap2::Mmap;
use crate::quant::gguf::{parse_header, GgufValue};

/// Resultado da compressão de embeddings de um modelo comercial para GRAPH 32-bit
pub struct EmbeddingCompressor {
    /// lex_id → embedding original (F32 normalizado, dimensão do modelo)
    pub embeddings: Vec<Vec<f32>>,
    /// lex_id → 32-bit GRAPH signature comprimido
    pub graph: HashMap<u32, u32>,
    /// lex_id → termo original do modelo (para debug)
    pub terms: Vec<String>,
    /// Dimensão original dos embeddings
    pub orig_dim: usize,
    /// Vocab size do modelo comercial
    pub vocab_size: usize,
}

impl EmbeddingCompressor {
    /// Carrega embeddings de um GGUF de modelo comercial (Llama, Qwen, Phi, etc.)
    /// Lê apenas o tensor `token_embd.weight` via mmap
    pub fn from_commercial_gguf(path: &str) -> Result<Self, String> {
        let file = File::open(path).map_err(|e| format!("abrir GGUF: {}", e))?;
        let mmap = unsafe { Mmap::map(&file).map_err(|e| format!("mmap: {}", e)) };
        let data = mmap.as_ref().map_err(|e| e.to_string())?;

        let header = parse_header(data).map_err(|e| format!("parse header: {:?}", e))?;

        // Detectar arquitetura
        let arch = header.kv_pairs.iter()
            .find(|kv| kv.key == "general.architecture")
            .and_then(|kv| if let GgufValue::String(s) = &kv.value { Some(s.as_str()) } else { None })
            .unwrap_or("unknown");

        let vocab_size = header.kv_pairs.iter()
            .find(|kv| kv.key == format!("{}.vocab_size", arch))
            .and_then(|kv| match &kv.value {
                GgufValue::U32(v) => Some(*v as usize),
                GgufValue::U64(v) => Some(*v as usize),
                _ => None,
            })
            .unwrap_or(0);

        let embd_dim = header.kv_pairs.iter()
            .find(|kv| kv.key == format!("{}.embedding_length", arch))
            .and_then(|kv| match &kv.value {
                GgufValue::U32(v) => Some(*v as usize),
                _ => None,
            })
            .unwrap_or(0);

        eprintln!("  Modelo detectado: {} (vocab={}, embd_dim={})", arch, vocab_size, embd_dim);

        // Encontrar tensor token_embd.weight
        let embd_info = header.tensor_infos.iter()
            .find(|t| t.name == "token_embd.weight")
            .ok_or("tensor token_embd.weight não encontrado")?;

        let start = header.data_offset as usize + embd_info.offset as usize;
        let numel: usize = embd_info.shape.iter().map(|&d| d as usize).product();
        let bytes_per_elem = match embd_info.quant_type {
            crate::quant::QuantType::F16 => 2,
            crate::quant::QuantType::F32 => 4,
            crate::quant::QuantType::Q8_0 => 1, // approximação
            _ => 2,
        };
        let end = start + numel * bytes_per_elem;
        if end > data.len() {
            return Err(format!("tensor excede tamanho do arquivo: {} > {}", end, data.len()));
        }

        // Ler e comprimir embeddings
        let mut graph: HashMap<u32, u32> = HashMap::new();
        let mut terms = Vec::new();

        // Ler raw bytes do tensor
        let raw = &data[start..end];

        for token_id in 0..vocab_size.min(128256) {
            let offset = token_id * embd_dim * bytes_per_elem;
            if offset + embd_dim * bytes_per_elem > raw.len() { break; }

            // Extrair embedding como F32
            let embd_f32 = match embd_info.quant_type {
                crate::quant::QuantType::F16 => {
                    let mut v = Vec::with_capacity(embd_dim);
                    for i in 0..embd_dim {
                        let bytes = [raw[offset + i*2], raw[offset + i*2 + 1]];
                        v.push(f16_to_f32(bytes));
                    }
                    v
                }
                crate::quant::QuantType::F32 => {
                    let mut v = Vec::with_capacity(embd_dim);
                    for i in 0..embd_dim {
                        let bytes = [raw[offset + i*4], raw[offset + i*4+1], raw[offset + i*4+2], raw[offset + i*4+3]];
                        v.push(f32::from_le_bytes(bytes));
                    }
                    v
                }
                _ => vec![0.0; embd_dim], // fallback
            };

            // Comprimir para 32-bit GRAPH via Random Indexing
            let signature = compress_embd_to_graph(&embd_f32, token_id as u32);
            graph.insert(token_id as u32, signature);
            terms.push(format!("token_{}", token_id));
        }

        eprintln!("  Comprimidos {} embeddings → GRAPH 32-bit", graph.len());

        Ok(Self {
            embeddings: Vec::new(), // não carrega tudo na memória
            graph,
            terms,
            orig_dim: embd_dim,
            vocab_size,
        })
    }

    /// Comprimir um embedding individual para 32-bit usando projetões aleatórias
    pub fn compress_single(embd: &[f32], seed: u32) -> u32 {
        compress_embd_to_graph(embd, seed)
    }
}

/// Compressão de embedding F32 de alta dimensão → 32-bit GRAPH
/// Usa projeções aleatórias (Random Indexing) + threshold binário
fn compress_embd_to_graph(embd: &[f32], seed: u32) -> u32 {
    let dim = embd.len();
    if dim == 0 { return 0; }

    // Gerar matrizes de projeção pseudo-aleatórias determinísticas
    // 32 projetões, cada uma com pesos ±1 distribuídos uniformemente
    let mut graph: u32 = 0;

    for bit in 0..32u32 {
        // Seed para cada bit: seed XOR bit position
        let bit_seed = seed.wrapping_mul(2654435761).wrapping_add(bit * 0x9E3779B9);
        let mut sum: f32 = 0.0;

        // Projeção: soma ponderada dos componentes do embedding
        // Usa subamostragem esparsa para eficiência: apenas ~sqrt(dim) componentes
        let sparsity = (dim as f32).sqrt() as usize;
        let stride = (dim / sparsity).max(1);

        for i in (0..dim).step_by(stride) {
            // Pseudo-aleatório: bit do componente XOR bit da seed
            let sign = if (i as u32).wrapping_mul(2654435761).wrapping_add(bit_seed) & 1 == 0 {
                1.0
            } else {
                -1.0
            };
            sum += embd[i] * sign;
        }

        // Threshold binário: soma positiva → bit 1
        if sum > 0.0 {
            graph |= 1 << bit;
        }
    }

    graph
}

/// Converter F16 bytes para F32
pub fn f16_to_f32(bytes: [u8; 2]) -> f32 {
    let bits = u16::from_le_bytes(bytes);
    let sign = ((bits >> 15) & 1) as f32;
    let exp = ((bits >> 10) & 0x1F) as i32;
    let mantissa = (bits & 0x3FF) as f32;

    if exp == 0 {
        if mantissa == 0.0 {
            return if sign == 0.0 { 0.0 } else { -0.0 };
        }
        // Denormalized
        let val = mantissa / 1024.0 * (2.0f32).powf(-14.0);
        return if sign == 0.0 { val } else { -val };
    }
    if exp == 31 {
        return if mantissa == 0.0 {
            if sign == 0.0 { f32::INFINITY } else { f32::NEG_INFINITY }
        } else {
            f32::NAN
        };
    }

    let val = (1.0 + mantissa / 1024.0) * (2.0f32).powf(exp as f32 - 15.0);
    if sign == 0.0 { val } else { -val }
}

/// Interpola entre GRAPH legado (Random Indexing) e GRAPH comprimido de LLM
/// Usado quando temos ambos os sinais disponíveis
pub fn interpolate_graphs(
    legacy_graph: &HashMap<u32, u32>,
    llm_graph: &HashMap<u32, u32>,
    llm_weight: f32, // 0.0 = só legado, 1.0 = só LLM
) -> HashMap<u32, u32> {
    let mut result = HashMap::new();

    // Para lex_ids que existem em ambos, interpolar bit a bit
    for (&lex_id, &legacy_sig) in legacy_graph {
        if let Some(&llm_sig) = llm_graph.get(&lex_id) {
            // Interpolação: para cada bit, decidir baseado no peso
            let mut merged: u32 = 0;
            for bit in 0..32u32 {
                let legacy_bit = (legacy_sig >> bit) & 1;
                let llm_bit = (llm_sig >> bit) & 1;
                // XOR ponderado: se peso do LLM > 0.5, preferir LLM
                let use_llm = (llm_weight > 0.5) == (llm_bit != 0);
                let use_legacy = (llm_weight <= 0.5) == (legacy_bit != 0);
                if use_llm || use_legacy {
                    merged |= 1 << bit;
                }
            }
            result.insert(lex_id, merged);
        } else {
            // Só tem legado
            result.insert(lex_id, legacy_sig);
        }
    }

    // Adicionar lex_ids que só existem no LLM
    for (&lex_id, &llm_sig) in llm_graph {
        if !result.contains_key(&lex_id) {
            result.insert(lex_id, llm_sig);
        }
    }

    result
}
