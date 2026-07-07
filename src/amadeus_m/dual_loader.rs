use std::collections::HashMap;
use std::fs::File;
use memmap2::Mmap;
use crate::quant::gguf::{parse_header, GgufValue};
use super::embedding_compressor::{EmbeddingCompressor, interpolate_graphs};

/// Loader dual-GGUF: mmap simultâneo de grammar + modelo comercial
pub struct DualGgufLoader {
    pub grammar_path: String,
    pub commercial_path: Option<String>,
}

/// Resultado do load dual
pub struct DualLoadResult {
    /// GRAPH do Amadeus (Random Indexing legado)
    pub grammar_graph: HashMap<u32, u32>,
    /// GRAPH comprimido do modelo comercial
    pub llm_graph: HashMap<u32, u32>,
    /// GRAPH interpolado (melhor dos dois mundos)
    pub merged_graph: HashMap<u32, u32>,
    /// Lexicon do compiler: form → (id, morph, class, style)
    pub lexicon: HashMap<String, (u32, u16, u8, u16)>,
    /// lex_id → classe morfológica
    pub lex_to_class: HashMap<u32, u8>,
    /// Metadados do modelo comercial
    pub commercial_arch: String,
    pub commercial_vocab_size: usize,
    pub commercial_embd_dim: usize,
}

impl DualGgufLoader {
    pub fn new(grammar_path: &str, commercial_path: Option<&str>) -> Self {
        Self {
            grammar_path: grammar_path.to_string(),
            commercial_path: commercial_path.map(|s| s.to_string()),
        }
    }

    /// Carrega ambos os GGUFs e retorna dados interpolados
    pub fn load(&self) -> Result<DualLoadResult, String> {
        eprintln!("╔══════════════════════════════════════════════════╗");
        eprintln!("║  DUAL GGUF LOADER — AMADEUS HYBRID              ║");
        eprintln!("╚══════════════════════════════════════════════════╝\n");

        // 1. Carregar grammar GGUF
        eprintln!("▸ Carregando grammar GGUF: {}", self.grammar_path);
        let grammar_data = std::fs::read(&self.grammar_path)
            .map_err(|e| format!("ler grammar: {}", e))?;
        let grammar_header = parse_header(&grammar_data)
            .map_err(|e| format!("parse grammar: {:?}", e))?;

        let mut grammar_graph = HashMap::new();
        let mut lexicon = HashMap::new();
        let mut lex_to_class = HashMap::new();

        // Ler graph.data
        if let Some(info) = grammar_header.tensor_infos.iter().find(|t| t.name == "graph.data") {
            let start = grammar_header.data_offset as usize + info.offset as usize;
            let numel: usize = info.shape.iter().map(|&d| d as usize).product();
            let byte_size = numel * 4;
            if start + byte_size <= grammar_data.len() {
                let raw = &grammar_data[start..start + byte_size];
                for chunk in raw.chunks_exact(8) {
                    let lex_id = u32::from_le_bytes(chunk[0..4].try_into().unwrap());
                    let sig = u32::from_le_bytes(chunk[4..8].try_into().unwrap());
                    grammar_graph.insert(lex_id, sig);
                }
            }
        }

        // Ler lex_to_class.data
        if let Some(info) = grammar_header.tensor_infos.iter().find(|t| t.name == "lex_to_class.data") {
            let start = grammar_header.data_offset as usize + info.offset as usize;
            let numel: usize = info.shape.iter().map(|&d| d as usize).product();
            let byte_size = numel * 4;
            if start + byte_size <= grammar_data.len() {
                let raw = &grammar_data[start..start + byte_size];
                for chunk in raw.chunks_exact(8) {
                    let lex_id = u32::from_le_bytes(chunk[0..4].try_into().unwrap());
                    let cls = u32::from_le_bytes(chunk[4..8].try_into().unwrap()) as u8;
                    lex_to_class.insert(lex_id, cls);
                }
            }
        }

        // Ler lexicon.forms
        if let Some(info) = grammar_header.tensor_infos.iter().find(|t| t.name == "lexicon.forms") {
            let start = grammar_header.data_offset as usize + info.offset as usize;
            let n_forms = info.shape[0] as usize;
            let row_size = 8 * 4; // 8 u32s per row
            if start + n_forms * row_size <= grammar_data.len() {
                let raw = &grammar_data[start..start + n_forms * row_size];
                for (i, chunk) in raw.chunks_exact(row_size).enumerate() {
                    let id = u32::from_le_bytes(chunk[0..4].try_into().unwrap());
                    let morph = u16::from_le_bytes(chunk[4..6].try_into().unwrap());
                    let cls = u8::from_le_bytes(chunk[8..9].try_into().unwrap());
                    let sty = u16::from_le_bytes(chunk[12..14].try_into().unwrap());
                    let form = format!("form_{}", i);
                    lexicon.insert(form, (id, morph, cls, sty));
                }
            }
        }

        eprintln!("  Grammar: {} GRAPH entries, {} lex entries, {} lex_to_class",
            grammar_graph.len(), lexicon.len(), lex_to_class.len());

        // 2. Carregar modelo comercial (se disponível)
        let (llm_graph, commercial_arch, commercial_vocab_size, commercial_embd_dim) =
            if let Some(ref path) = self.commercial_path {
                eprintln!("\n▸ Carregando modelo comercial: {}", path);
                let compressor = EmbeddingCompressor::from_commercial_gguf(path)?;
                (compressor.graph, "llama".into(), compressor.vocab_size, compressor.orig_dim)
            } else {
                eprintln!("\n  (sem modelo comercial — usando GRAPH legado)");
                (HashMap::new(), "none".into(), 0, 0)
            };

        // 3. Interpolar GRAPHs
        let merged = if !llm_graph.is_empty() {
            eprintln!("\n▸ Interpolando GRAPHs (legado + LLM)...");
            let merged = interpolate_graphs(&grammar_graph, &llm_graph, 0.6);
            eprintln!("  Resultado: {} GRAPH entries", merged.len());
            merged
        } else {
            grammar_graph.clone()
        };

        Ok(DualLoadResult {
            grammar_graph,
            llm_graph,
            merged_graph: merged,
            lexicon,
            lex_to_class,
            commercial_arch,
            commercial_vocab_size,
            commercial_embd_dim,
        })
    }
}

/// mmap Reader para tensors específicos de um GGUF
/// Permite acessar tensores sem carregar o arquivo inteiro
pub struct GgufMmapReader {
    #[allow(dead_code)]
    file: File,
    #[allow(dead_code)]
    mmap: Mmap,
    data: Vec<u8>,
    #[allow(dead_code)]
    header: crate::quant::gguf::GgufHeader,
}

impl GgufMmapReader {
    pub fn open(path: &str) -> Result<Self, String> {
        let file = File::open(path).map_err(|e| format!("abrir: {}", e))?;
        let mmap = unsafe { Mmap::map(&file).map_err(|e| format!("mmap: {}", e))? };
        let data = mmap.to_vec(); // copia para acesso seguro
        let header = parse_header(&data).map_err(|e| format!("parse: {:?}", e))?;
        Ok(Self { file, mmap, data, header })
    }

    /// Ler tensor F32 por nome
    pub fn read_tensor_f32(&self, name: &str) -> Option<Vec<f32>> {
        let info = self.header.tensor_infos.iter().find(|t| t.name == name)?;
        let start = self.header.data_offset as usize + info.offset as usize;
        let numel: usize = info.shape.iter().map(|&d| d as usize).product();

        match info.quant_type {
            crate::quant::QuantType::F32 => {
                let byte_size = numel * 4;
                if start + byte_size > self.data.len() { return None; }
                let raw = &self.data[start..start + byte_size];
                Some(raw.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
            }
            crate::quant::QuantType::F16 => {
                let byte_size = numel * 2;
                if start + byte_size > self.data.len() { return None; }
                let raw = &self.data[start..start + byte_size];
                Some(raw.chunks_exact(2).map(|c| {
                    let bits = u16::from_le_bytes(c.try_into().unwrap());
                    super::embedding_compressor::f16_to_f32([c[0], c[1]])
                }).collect())
            }
            _ => None,
        }
    }

    /// Ler tensor bruto (bytes) por nome
    pub fn read_tensor_raw(&self, name: &str) -> Option<&[u8]> {
        let info = self.header.tensor_infos.iter().find(|t| t.name == name)?;
        let start = self.header.data_offset as usize + info.offset as usize;
        let numel: usize = info.shape.iter().map(|&d| d as usize).product();
        let bytes_per = match info.quant_type {
            crate::quant::QuantType::F32 => 4,
            crate::quant::QuantType::F16 => 2,
            crate::quant::QuantType::Q8_0 => 1,
            _ => 4,
        };
        let byte_size = numel * bytes_per;
        if start + byte_size > self.data.len() { return None; }
        Some(&self.data[start..start + byte_size])
    }

    /// Listar todos os tensores disponíveis
    pub fn list_tensors(&self) -> Vec<(&str, Vec<u64>)> {
        self.header.tensor_infos.iter()
            .map(|t| (t.name.as_str(), t.shape.clone()))
            .collect()
    }
}
