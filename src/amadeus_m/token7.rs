use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Token7 {
    pub lex: u32,
    pub morph: u16,
    pub syn_off: i16,
    pub syn_func: u8,
    pub orth: u8,
    pub punct: u8,
    pub style: u16,
    pub graph: u32,
}

impl Token7 {
    pub fn new(lex: u32, morph: u16) -> Self {
        Self { lex, morph, syn_off: 0, syn_func: 0, orth: 0, punct: 0, style: 0, graph: 0 }
    }

    pub fn morph_class(&self) -> u8 { (self.morph & 0x07) as u8 }
    pub fn morph_gender(&self) -> u8 { ((self.morph >> 3) & 0x01) as u8 }
    pub fn morph_number(&self) -> u8 { ((self.morph >> 4) & 0x01) as u8 }
    pub fn morph_person(&self) -> u8 { ((self.morph >> 8) & 0x07) as u8 }
    pub fn morph_tense(&self) -> u8 { ((self.morph >> 5) & 0x07) as u8 }
    pub fn morph_detail(&self) -> u16 { self.morph & !0x07 }

    pub fn with_syn(self, off: i16, func: u8) -> Self {
        Self { syn_off: off, syn_func: func, ..self }
    }
    pub fn with_punct(self, p: u8) -> Self {
        Self { punct: p, ..self }
    }
    pub fn with_style(self, s: u16) -> Self {
        Self { style: s, ..self }
    }
    pub fn with_graph(self, g: u32) -> Self {
        Self { graph: g, ..self }
    }

    pub fn to_flat(&self) -> Vec<u32> {
        vec![self.lex, self.morph as u32, self.syn_off as u32,
             self.syn_func as u32, self.orth as u32, self.punct as u32,
             self.style as u32, self.graph]
    }

    pub fn from_flat(flat: &[u32]) -> Self {
        Self {
            lex: flat[0], morph: flat[1] as u16, syn_off: flat[2] as i16,
            syn_func: flat[3] as u8, orth: flat[4] as u8, punct: flat[5] as u8,
            style: flat[6] as u16, graph: flat[7],
        }
    }

    pub fn flatten_all(tokens: &[Token7]) -> Vec<u32> {
        let mut v = Vec::with_capacity(tokens.len() * 8);
        for t in tokens { v.extend(t.to_flat()); }
        v
    }

    pub fn unflatten_all(flat: &[u32]) -> Vec<Token7> {
        flat.chunks_exact(8).map(Token7::from_flat).collect()
    }
}

pub const CLASS_SUBSTANTIVO: u8 = 0;
pub const CLASS_VERBO: u8       = 1;
pub const CLASS_ADJETIVO: u8    = 2;
pub const CLASS_ARTIGO: u8      = 3;
pub const CLASS_PREPOSICAO: u8  = 4;
pub const CLASS_PONTUACAO: u8   = 5;
pub const CLASS_OUTRO: u8       = 6;

pub const SYN_ROOT: u8    = 4;
pub const SYN_SUJEITO: u8 = 0;
pub const SYN_OD: u8      = 1;
pub const SYN_OI: u8      = 2;
pub const SYN_ADV: u8     = 3;
pub const SYN_ADN: u8     = 5;
pub const SYN_PREP: u8    = 6;
pub const SYN_PRED: u8    = 7;

pub fn morph_from_class(class: u8, gender: u8, number: u8, person: u8, tense: u8) -> u16 {
    (class as u16) | ((gender as u16) << 3) | ((number as u16) << 4)
    | ((tense as u16) << 5) | ((person as u16) << 8)
}

pub const PUNCT_PERIOD: u8      = 1;
pub const PUNCT_COMMA: u8       = 2;
pub const PUNCT_SEMICOLON: u8   = 3;
pub const PUNCT_COLON: u8       = 4;
pub const PUNCT_DASH: u8        = 5;
pub const PUNCT_ELLIPSIS: u8    = 6;
pub const PUNCT_QUESTION: u8    = 7;
pub const PUNCT_EXCLAMATION: u8 = 8;

pub fn decompile_tokens(tokens: &[Token7], lexicon: &HashMap<String, (u32, u16, u8, u16)>) -> String {
    let mut out = String::new();
    for t in tokens {
        if t.morph_class() == CLASS_PONTUACAO {
            let ch = match t.punct {
                1 => ".", 2 => ",", 3 => ";", 4 => ":", 5 => "—", 6 => "...",
                7 => "?", 8 => "!", _ => "",
            };
            if ch.is_empty() && t.lex != 0 {
                if let Some((form, _)) = lexicon.iter().find(|(_, val)| val.0 == t.lex) {
                    out.push_str(form);
                }
            } else if !out.is_empty() && !out.ends_with(' ') {
                out.push_str(ch);
            } else {
                out.push_str(ch);
            }
        } else {
            if !out.is_empty() && !out.ends_with(' ') { out.push(' '); }
            if let Some((form, _)) = lexicon.iter().find(|(_, val)| val.0 == t.lex) {
                out.push_str(form);
            } else {
                out.push_str(&format!("[{}]", t.lex));
            }
        }
    }
    out
}
