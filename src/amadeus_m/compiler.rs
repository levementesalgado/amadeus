use std::collections::HashMap;
use std::fs;
use std::path::Path;
use crate::amadeus_m::token7::{
    Token7, morph_from_class,
    CLASS_SUBSTANTIVO, CLASS_VERBO, CLASS_ADJETIVO, CLASS_ARTIGO,
    CLASS_PREPOSICAO, CLASS_PONTUACAO, CLASS_OUTRO,
    SYN_ROOT, SYN_SUJEITO,
};

fn pos_to_class(s: &str) -> u8 {
    match s.to_uppercase().as_str() {
        "ART" | "ARTIGO" => CLASS_ARTIGO,
        "SUBST" | "SUBSTANTIVO" => CLASS_SUBSTANTIVO,
        "ADJ" | "ADJETIVO" => CLASS_ADJETIVO,
        "VERB" | "VERBO" => CLASS_VERBO,
        "ADV" | "ADVERBIO" => CLASS_OUTRO,
        "PREP" | "PREPOSICAO" => CLASS_PREPOSICAO,
        "PRON" | "PRONOME" => CLASS_OUTRO,
        "CONJ" | "CONJUNCAO" => CLASS_OUTRO,
        _ => CLASS_OUTRO,
    }
}

fn old_bits_to_morph(bits_str: &str, class: u8) -> u16 {
    let bits = u128::from_str_radix(bits_str.trim().strip_prefix("0x").unwrap_or(bits_str.trim()), 16).unwrap_or(0);
    let gender = if bits & 0x20 != 0 { 1 } else { 0 };
    let number = if bits & 0x80 != 0 { 1 } else { 0 };
    let person = ((bits >> 8) & 0x03) as u8;
    let tense = ((bits >> 10) & 0x07) as u8;
    morph_from_class(class, gender, number, person, tense)
}

fn punct_from_str(s: &str) -> u8 {
    match s {
        "." => 1, "," => 2, ";" => 3, ":" => 4, "—" => 5, "-" => 5,
        "..." => 6, "?" => 7, "!" => 8, "(" => 9, ")" => 10,
        _ => 0,
    }
}

pub struct Compiler {
    pub lexicon: HashMap<String, (u32, u16, u8, u16)>, // form → (id, morph, class, style)
    pub roots: Vec<String>,
    pub next_id: u32,
    pub graph: HashMap<u32, u32>, // lex → assinatura semântica (random indexing)
    base_path: String,
}

fn word_style(id: u32, class: u8) -> u16 {
    if class == CLASS_PONTUACAO { return 0; }
    // 3 registros: 0=neutro, 1=formal, 2=informal
    match id.wrapping_mul(2654435761) % 3 {
        0 => 0,
        1 => 1,
        _ => 2,
    }
}

impl Compiler {
    pub fn new(underworld_path: &str) -> Self {
        let base = format!("{}/lingua/raizes", underworld_path);
        let mut c = Self {
            lexicon: HashMap::new(),
            roots: vec![String::new()],
            next_id: 1,
            graph: HashMap::new(),
            base_path: base,
        };

        for p in &[",", ".", "!", "?", ";", ":", "—", "-", "\"", "(", ")", "..."] {
            let morph = morph_from_class(CLASS_PONTUACAO, 0, 0, 0, 0);
            c.lexicon.insert(p.to_string(), (0, morph, CLASS_PONTUACAO, 0));
        }

        c.load_all();
        c
    }

    fn load_all(&mut self) {
        let files = [
            "substantivos.txt", "verbos.txt", "adjetivos.txt",
            "adverbios.txt", "pronomes.txt", "preposicoes.txt",
            "artigos.txt", "conjuncoes.txt",
        ];
        for fname in &files { self.load_roots(fname); }

        let exc_path = format!("{}/excecoes.txt", self.base_path);
        if Path::new(&exc_path).exists() {
            if let Ok(content) = fs::read_to_string(&exc_path) {
                for line in content.lines() {
                    let line = line.trim();
                    if line.is_empty() || line.starts_with('#') { continue; }
                    let parts: Vec<&str> = line.splitn(4, ',').collect();
                    if parts.len() < 4 { continue; }
                    let form = parts[1].trim().to_string();
                    let root_id: u32 = parts[2].trim().parse().unwrap_or(0);
                    let bits_str = parts[3].trim();
                    let class = CLASS_OUTRO;
                    let morph = old_bits_to_morph(bits_str, class);
                    let style = word_style(root_id, class);
                    self.lexicon.insert(form, (root_id, morph, class, style));
                }
            }
        }
    }

    fn load_roots(&mut self, fname: &str) {
        let path = format!("{}/{}", self.base_path, fname);
        if !Path::new(&path).exists() { return; }
        if let Ok(content) = fs::read_to_string(&path) {
            for line in content.lines() {
                let line = line.trim();
                if line.is_empty() || line.starts_with('#') { continue; }
                let parts: Vec<&str> = line.splitn(4, ',').collect();
                if parts.len() < 4 { continue; }
                let id: u32 = parts[0].trim().parse().unwrap_or(0);
                let raiz = parts[1].trim().to_string();
                let class = pos_to_class(parts[2].trim());
                let bits_str = parts[3].trim();
                let morph = old_bits_to_morph(bits_str, class);

                if id as usize >= self.roots.len() {
                    self.roots.resize(id as usize + 1, String::new());
                }
                self.roots[id as usize] = raiz.clone();
                if !self.lexicon.contains_key(&raiz) {
                    let style = word_style(id, class);
                    self.lexicon.insert(raiz, (id, morph, class, style));
                }
                if id >= self.next_id { self.next_id = id + 1; }
            }
        }
    }

    pub fn compile(&mut self, text: &str) -> Vec<Token7> {
        let mut tokens = Vec::new();
        let words = self.tokenize(text);
        for w in words {
            let lower = w.to_lowercase();
            if let Some(&(id, morph, class, style)) = self.lexicon.get(&lower) {
                let punct = if class == CLASS_PONTUACAO { punct_from_str(&w) } else { 0 };
                let g = self.graph.get(&id).copied().unwrap_or(0);
                tokens.push(Token7::new(id, morph).with_punct(punct).with_graph(g).with_style(style));
            } else {
                let (id, morph, _class, style) = self.infer_unknown(&w);
                let g = self.graph.get(&id).copied().unwrap_or(0);
                tokens.push(Token7::new(id, morph).with_graph(g).with_style(style));
            }
        }
        tokens
    }

    pub fn decompile(&self, tokens: &[Token7]) -> String {
        let mut out = String::new();
        for t in tokens {
            if t.morph_class() == CLASS_PONTUACAO {
                let ch = match t.punct {
                    1 => ".", 2 => ",", 3 => ";", 4 => ":", 5 => "—", 6 => "...",
                    7 => "?", 8 => "!", _ => "",
                };
                if !ch.is_empty() {
                    out.push_str(ch);
                } else if t.lex != 0 {
                    if let Some((form, _)) = self.lexicon.iter().find(|(_, val)| val.0 == t.lex) {
                        out.push_str(form);
                    }
                }
            } else {
                if !out.is_empty() && !out.ends_with(' ') { out.push(' '); }
                if let Some((form, _)) = self.lexicon.iter().find(|(_, val)| val.0 == t.lex) {
                    out.push_str(form);
                } else {
                    let idx = t.lex as usize;
                    if idx < self.roots.len() && !self.roots[idx].is_empty() {
                        out.push_str(&self.roots[idx]);
                    } else {
                        out.push_str(&format!("[{}]", t.lex));
                    }
                }
            }
        }
        out
    }

    fn tokenize(&self, text: &str) -> Vec<String> {
        let mut words = Vec::new();
        let mut current = String::new();
        for ch in text.chars() {
            if ch.is_alphabetic() || ch == '\'' {
                current.push(ch);
            } else {
                if !current.is_empty() {
                    words.push(current.clone());
                    current.clear();
                }
                if !ch.is_whitespace() {
                    words.push(ch.to_string());
                }
            }
        }
        if !current.is_empty() { words.push(current); }
        words
    }

    pub fn save_state(&self, path: &str) -> std::io::Result<()> {
        let mut data = Vec::new();
        data.extend_from_slice(&self.next_id.to_le_bytes());
        let n_roots = self.roots.len() as u32;
        data.extend_from_slice(&n_roots.to_le_bytes());
        for root in &self.roots {
            let bytes = root.as_bytes();
            data.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
            data.extend_from_slice(bytes);
        }
        let n_lex = self.lexicon.len() as u32;
        data.extend_from_slice(&n_lex.to_le_bytes());
        for (form, &(id, morph, class, style)) in &self.lexicon {
            let bytes = form.as_bytes();
            data.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
            data.extend_from_slice(bytes);
            data.extend_from_slice(&id.to_le_bytes());
            data.extend_from_slice(&morph.to_le_bytes());
            data.extend_from_slice(&class.to_le_bytes());
            data.extend_from_slice(&style.to_le_bytes());
        }
        // graph
        let n_graph = self.graph.len() as u32;
        data.extend_from_slice(&n_graph.to_le_bytes());
        for (&k, &v) in &self.graph {
            data.extend_from_slice(&k.to_le_bytes());
            data.extend_from_slice(&v.to_le_bytes());
        }
        fs::write(path, &data)
    }

    pub fn load_state(&mut self, path: &str) -> std::io::Result<()> {
        let data = fs::read(path)?;
        let mut off = 0;
        self.next_id = u32::from_le_bytes(data[off..off+4].try_into().unwrap());
        off += 4;
        let n_roots = u32::from_le_bytes(data[off..off+4].try_into().unwrap()) as usize;
        off += 4;
        self.roots.clear();
        for _ in 0..n_roots {
            let len = u32::from_le_bytes(data[off..off+4].try_into().unwrap()) as usize;
            off += 4;
            let s = String::from_utf8(data[off..off+len].to_vec()).unwrap_or_default();
            off += len;
            self.roots.push(s);
        }
        let n_lex = u32::from_le_bytes(data[off..off+4].try_into().unwrap()) as usize;
        off += 4;
        self.lexicon.clear();
        for _ in 0..n_lex {
            let len = u32::from_le_bytes(data[off..off+4].try_into().unwrap()) as usize;
            off += 4;
            let form = String::from_utf8(data[off..off+len].to_vec()).unwrap_or_default();
            off += len;
            let id = u32::from_le_bytes(data[off..off+4].try_into().unwrap());
            off += 4;
            let morph = u16::from_le_bytes(data[off..off+2].try_into().unwrap());
            off += 2;
            let class = u8::from_le_bytes(data[off..off+1].try_into().unwrap());
            off += 1;
            let style = if off + 2 <= data.len() {
                u16::from_le_bytes(data[off..off+2].try_into().unwrap_or([0;2]))
            } else { 0 };
            off += 2;
            self.lexicon.insert(form, (id, morph, class, style));
        }
        // graph
        self.graph.clear();
        if off + 4 <= data.len() {
            let n_graph = u32::from_le_bytes(data[off..off+4].try_into().unwrap_or([0;4])) as usize;
            off += 4;
            for _ in 0..n_graph {
                if off + 8 > data.len() { break; }
                let k = u32::from_le_bytes(data[off..off+4].try_into().unwrap()); off += 4;
                let v = u32::from_le_bytes(data[off..off+4].try_into().unwrap()); off += 4;
                self.graph.insert(k, v);
            }
        }
        Ok(())
    }

    /// Constrói assinaturas semânticas via random indexing.
    /// Cada lex ganha um u32 onde bits similares = contextos similares.
    pub fn build_graph(&mut self, tokens: &[Token7], window: usize) {
        let mut rng = fastrand::Rng::with_seed(42);
        let mut base: HashMap<u32, u32> = HashMap::new();
        for t in tokens {
            base.entry(t.lex).or_insert_with(|| rng.u32(..));
        }
        let mut sig: HashMap<u32, u32> = base.clone();
        for i in 0..tokens.len() {
            let lex = tokens[i].lex;
            if lex == 0 { continue; }
            let start = if i > window { i - window } else { 0 };
            let end = (i + window + 1).min(tokens.len());
            for j in start..end {
                if i == j { continue; }
                let ctx = tokens[j].lex;
                if ctx == 0 { continue; }
                let offset = if j < i { (i - j) as u32 } else { (j - i) as u32 };
                if let Some(g) = sig.get_mut(&lex) {
                    if let Some(&b) = base.get(&ctx) {
                        *g ^= b.rotate_left(offset);
                    }
                }
            }
        }
        self.graph = sig;
    }

    fn infer_unknown(&mut self, word: &str) -> (u32, u16, u8, u16) {
        let lower = word.to_lowercase();
        let class = if lower.ends_with('a') || lower.ends_with("ção") || lower.ends_with("dade") {
            CLASS_SUBSTANTIVO
        } else if lower.ends_with('o') || lower.ends_with("or") {
            CLASS_SUBSTANTIVO
        } else if lower.ends_with("mente") {
            CLASS_OUTRO
        } else {
            CLASS_SUBSTANTIVO
        };
        let gender = if lower.ends_with('a') { 1 } else { 0 };
        let morph = morph_from_class(class, gender, 0, 0, 0);
        let id = self.next_id;
        self.next_id += 1;
        let style = word_style(id, class);
        if id as usize >= self.roots.len() {
            self.roots.resize(id as usize + 1, String::new());
        }
        self.roots[id as usize] = lower.clone();
        self.lexicon.insert(lower, (id, morph, class, style));
        (id, morph, class, style)
    }
}

// Keep old constants for backward compat — they map to new values
pub const POS_ARTIGO: u8 = CLASS_ARTIGO;
pub const POS_SUBSTANTIVO: u8 = CLASS_SUBSTANTIVO;
pub const POS_ADJETIVO: u8 = CLASS_ADJETIVO;
pub const POS_VERBO: u8 = CLASS_VERBO;
pub const POS_ADVERBIO: u8 = CLASS_OUTRO;
pub const POS_PREPOSICAO: u8 = CLASS_PREPOSICAO;
pub const POS_PRONOME: u8 = CLASS_OUTRO;
pub const POS_CONJUNCAO: u8 = CLASS_OUTRO;
pub const POS_PONTUACAO: u8 = CLASS_PONTUACAO;
pub const SYN_NUC: u8 = SYN_ROOT;
pub const SYN_S: u8 = SYN_SUJEITO;
