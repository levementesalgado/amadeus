use std::collections::HashMap;

/// Vocabulário de afixos portugueses
/// Cada afixo é mapeado para um conjunto de bits (32 dims)
/// Palavras são representadas como XOR dos bits dos seus afixos

/// Prefixos comuns do português
const PORTUGUESE_PREFIXES: &[&str] = &[
    "a", "ad", "anti", "auto", "bi", "co", "com", "con", "contra",
    "de", "des", "dis", "en", "ex", "extra", "hi", "hiper",
    "im", "in", "inter", "ir", "macro", "micro", "mono", "multi",
    "neo", "pós", "prê", "pré", "pro", "re", "retro", "semi",
    "sub", "super", "trans", "tri", "uni", "vice",
    // Prefixos verbais
    "des", "ent", "enf", "enr", "env",
    // Prefixos nominais
    "contra", "dest", "entre", "sobre",
];

/// Sufixos comuns do português
const PORTUGUESE_SUFFIXES: &[&str] = &[
    // Sufixos nominais
    "ção", "ções", "dade", "dades", "ismo", "ismos", "ista", "istas",
    "ância", "ânncias", "ência", "ências", "ário", "ários", "eiro", "eira",
    "agem", "agens", "al", "ais", "el", "eis", "ol", "óis",
    "il", "is", "ul", "uis",
    // Sufixos verbais
    "ar", "ars", "er", "ers", "ir", "irs", "or", "ores",
    "ando", "endo", "indo", "ondo",
    "ado", "ados", "ida", "idas", "ido", "idos",
    // Sufixos adjetivais
    "oso", "osa", "osos", "osas", "ível", "íveis", "ável", "áveis",
    "al", "ais", "el", "eis", "il", "is",
    "ento", "enta", "entos", "mento", "mentos",
    // Sufixos adverbiais
    "mente", "mentes",
    // Sufixos diminutivos/aumentativos
    "inho", "inha", "inhos", "inhas",
    "ão", "ãos", "ote", "otes",
    // Sufixos de grau
    "íssimo", "íssima", "íssimos", "íssimas",
];

/// Raízes comuns (verbos frequentes, substantivos frequentes)
const COMMON_ROOTS: &[&str] = &[
    // Verbos comuns
    "ser", "estar", "ter", "haver", "fazer", "dizer", "ir", "vir",
    "dar", "ver", "saber", "poder", "querer", "ficar", "passar",
    "achar", "trazer", "colocar", "começar", "voltar", "partir",
    "entrar", "sair", "olhar", "andar", "correr", "sentir",
    // Substantivos comuns
    "ano", "dia", "tempo", "vez", "casa", "vida", "mundo", "terra",
    "homem", "mulher", "criança", "pessoa", "gente", "mão",
    "olho", "cabeça", "corpo", "coração", "palavra", "letra",
    // Adjetivos comuns
    "bom", "mau", "grande", "pequeno", "novo", "velho",
    "primeiro", "último", "melhor", "pior",
];

pub struct AffixVocabulary {
    pub prefix_to_id: HashMap<String, u32>,
    pub suffix_to_id: HashMap<String, u32>,
    pub root_to_id: HashMap<String, u32>,
    pub id_to_affix: HashMap<u32, String>,
    pub affix_bits: HashMap<u32, u32>, // id → 32-bit embedding
    pub next_id: u32,
}

impl AffixVocabulary {
    pub fn new() -> Self {
        let mut v = Self {
            prefix_to_id: HashMap::new(),
            suffix_to_id: HashMap::new(),
            root_to_id: HashMap::new(),
            id_to_affix: HashMap::new(),
            affix_bits: HashMap::new(),
            next_id: 1,
        };

        // Registrar afixos
        for prefix in PORTUGUESE_PREFIXES {
            v.register_affix(prefix, "prefix");
        }
        for suffix in PORTUGUESE_SUFFIXES {
            v.register_affix(suffix, "suffix");
        }
        for root in COMMON_ROOTS {
            v.register_affix(root, "root");
        }

        // Gerar bits determinísticos para cada afixo
        v.generate_affix_bits();

        v
    }

    fn register_affix(&mut self, affix: &str, _kind: &str) {
        if self.prefix_to_id.contains_key(affix) || self.suffix_to_id.contains_key(affix) || self.root_to_id.contains_key(affix) {
            return;
        }

        let id = self.next_id;
        self.next_id += 1;
        self.id_to_affix.insert(id, affix.to_string());

        match _kind {
            "prefix" => { self.prefix_to_id.insert(affix.to_string(), id); }
            "suffix" => { self.suffix_to_id.insert(affix.to_string(), id); }
            "root" => { self.root_to_id.insert(affix.to_string(), id); }
            _ => {}
        }
    }

    fn generate_affix_bits(&mut self) {
        // Gerar bits determinísticos usando hash do nome do afixo
        for (&id, affix) in &self.id_to_affix.clone() {
            let bits = self.affix_hash(affix);
            self.affix_bits.insert(id, bits);
        }
    }

    /// Hash determinístico de um afixo para 32 bits
    fn affix_hash(&self, affix: &str) -> u32 {
        let mut hash: u32 = 0x811c9dc5; // FNV offset basis
        for byte in affix.bytes() {
            hash ^= byte as u32;
            hash = hash.wrapping_mul(0x01000193); // FNV prime
        }

        // Em seguida, criar bits mais significativos baseados no conteúdo
        let mut bits: u32 = 0;

        // Bits 0-7: comprimento do afixo (0-31)
        bits |= (affix.len() as u32 & 0x1F) << 0;

        // Bits 8-15: primeira letra (a-z → 0-25)
        if let Some(first) = affix.chars().next() {
            bits |= ((first as u32 - 'a' as u32) & 0x1F) << 8;
        }

        // Bits 16-23: última letra
        if let Some(last) = affix.chars().last() {
            bits |= ((last as u32 - 'a' as u32) & 0x1F) << 16;
        }

        // Bits 24-31: padrão de vogais/consoantes
        let vowel_count = affix.chars().filter(|c| "aeiou".contains(*c)).count() as u32;
        bits |= (vowel_count & 0x0F) << 24;

        // XOR com hash FNV para espalhar melhor
        bits ^ hash
    }

    /// Decompor uma palavra em afixos
    /// Retorna (prefix_id, root_id, suffix_id)
    pub fn decompose(&self, word: &str) -> (Option<u32>, Option<u32>, Option<u32>) {
        let word_lower = word.to_lowercase();
        let word_chars: Vec<char> = word_lower.chars().collect();
        let word_len = word_chars.len();

        // Tentar encontrar o prefixo mais longo
        let mut best_prefix: Option<u32> = None;
        let mut best_prefix_len = 0;
        for (prefix, &id) in &self.prefix_to_id {
            let prefix_chars: Vec<char> = prefix.chars().collect();
            let prefix_len = prefix_chars.len();
            if prefix_len < word_len && word_chars[..prefix_len] == prefix_chars[..] {
                if prefix_len > best_prefix_len {
                    best_prefix = Some(id);
                    best_prefix_len = prefix_len;
                }
            }
        }

        // Tentar encontrar o sufixo mais longo (sem sobrepor com o prefixo)
        let mut best_suffix: Option<u32> = None;
        let mut best_suffix_len = 0;
        for (suffix, &id) in &self.suffix_to_id {
            let suffix_chars: Vec<char> = suffix.chars().collect();
            let suffix_len = suffix_chars.len();
            // Verificar se o sufixo cabe (não sobrepor com prefixo)
            if suffix_len < word_len && best_prefix_len + suffix_len < word_len {
                if word_chars[word_len - suffix_len..] == suffix_chars[..] {
                    if suffix_len > best_suffix_len {
                        best_suffix = Some(id);
                        best_suffix_len = suffix_len;
                    }
                }
            }
        }

        // O que sobra é a raiz
        let root_len = word_len - best_prefix_len - best_suffix_len;

        let best_root = if root_len > 0 {
            let root_str: String = word_chars[best_prefix_len..best_prefix_len + root_len].iter().collect();
            self.root_to_id.get(&root_str).copied()
        } else {
            None
        };

        (best_prefix, best_root, best_suffix)
    }

    /// Calcular embedding de uma palavra a partir dos seus afixos
    /// Usa XOR dos bits dos afixos
    pub fn word_embedding(&self, word: &str) -> u32 {
        let (prefix, root, suffix) = self.decompose(word);

        let mut embedding: u32 = 0;

        if let Some(id) = prefix {
            if let Some(&bits) = self.affix_bits.get(&id) {
                embedding ^= bits;
            }
        }
        if let Some(id) = root {
            if let Some(&bits) = self.affix_bits.get(&id) {
                embedding ^= bits;
            }
        }
        if let Some(id) = suffix {
            if let Some(&bits) = self.affix_bits.get(&id) {
                embedding ^= bits;
            }
        }

        // Se não encontrou nenhum afixo, usar hash da palavra inteira
        if embedding == 0 {
            embedding = self.affix_hash(word);
        }

        embedding
    }

    /// Calcular similaridade entre duas palavras baseada nos afixos
    /// Retorna valor entre 0.0 (diferente) e 1.0 (idêntico)
    pub fn morphological_similarity(&self, word1: &str, word2: &str) -> f32 {
        let (p1, r1, s1) = self.decompose(word1);
        let (p2, r2, s2) = self.decompose(word2);

        let mut score = 0.0f32;
        let mut total = 0.0f32;

        // Mesmo prefixo: +0.3
        if p1 == p2 && p1.is_some() {
            score += 0.3;
        }
        total += 0.3;

        // Mesma raiz: +0.4
        if r1 == r2 && r1.is_some() {
            score += 0.4;
        }
        total += 0.4;

        // Mesmo sufixo: +0.3
        if s1 == s2 && s1.is_some() {
            score += 0.3;
        }
        total += 0.3;

        score / total
    }

    /// Encontrar palavras morfologicamente similares
    pub fn find_similar(&self, word: &str, vocabulary: &[String], k: usize) -> Vec<(String, f32)> {
        let mut candidates: Vec<(String, f32)> = vocabulary.iter()
            .filter(|w| *w != word)
            .map(|w| (w.clone(), self.morphological_similarity(word, w)))
            .filter(|(_, sim)| *sim > 0.0)
            .collect();

        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        candidates.truncate(k);
        candidates
    }

    /// Contar afixos únicos
    pub fn stats(&self) -> AffixStats {
        AffixStats {
            n_prefixes: self.prefix_to_id.len(),
            n_suffixes: self.suffix_to_id.len(),
            n_roots: self.root_to_id.len(),
            n_total: self.next_id as usize - 1,
        }
    }
}

pub struct AffixStats {
    pub n_prefixes: usize,
    pub n_suffixes: usize,
    pub n_roots: usize,
    pub n_total: usize,
}

impl std::fmt::Display for AffixStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Afixos: {} prefixos + {} sufixos + {} raízes = {} total",
            self.n_prefixes, self.n_suffixes, self.n_roots, self.n_total
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decompose() {
        let vocab = AffixVocabulary::new();

        // "casas" = cas (root) + as (suffix)
        let (prefix, root, suffix) = vocab.decompose("casas");
        println!("casas: prefix={:?}, root={:?}, suffix={:?}", prefix, root, suffix);

        // "bonito" = bon (root) + ito (suffix)
        let (prefix, root, suffix) = vocab.decompose("bonito");
        println!("bonito: prefix={:?}, root={:?}, suffix={:?}", prefix, root, suffix);

        // "invisível" = in (prefix) + vis (root) +ível (suffix)
        let (prefix, root, suffix) = vocab.decompose("invisível");
        println!("invisível: prefix={:?}, root={:?}, suffix={:?}", prefix, root, suffix);
    }

    #[test]
    fn test_similarity() {
        let vocab = AffixVocabulary::new();

        let sim = vocab.morphological_similarity("casa", "casas");
        println!("casa vs casas: {}", sim);

        let sim = vocab.morphological_similarity("bonito", "bonita");
        println!("bonito vs bonita: {}", sim);

        let sim = vocab.morphological_similarity("invisível", "visível");
        println!("invisível vs visível: {}", sim);

        let sim = vocab.morphological_similarity("casa", "carro");
        println!("casa vs carro: {}", sim);
    }
}
