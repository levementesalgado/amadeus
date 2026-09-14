use std::collections::HashMap;

/// Vocabulário morfológico expandido do português
/// ~600 afixos + ~500 raízes = ~1100 componentes
/// Cada componente → 32-bit hash → palavra = XOR dos bits

// ═══════════════════════════════════════════════════════
// PREFIXOS (~80)
// ═══════════════════════════════════════════════════════

const PREFIXES: &[&str] = &[
    // Negação / oposição
    "anti", "contra", "des", "dis", "in", "im", "ir", "non",
    // Direção / movimento
    "ad", "co", "com", "con", "de", "en", "ent", "ex",
    "inter", "intra", "meta", "para", "per", "por", "pre",
    "pro", "re", "retro", "semi", "sub", "super", "supra",
    "trans", "ultra",
    // Tamanho / intensidade
    "hiper", "macro", "micro", "mini", "multi", "omni",
    "poli", "proto", "pseudo",
    // Tempo
    "pós", "pré", "ante", "neo", "paleo",
    // Número
    "bi", "di", "mono", "poli", "tri", "uni", "tetra", "penta",
    "hexa", "hepta", "octa", "ennea", "deca",
    // Origem / relação
    "auto", "eu", "hetero", "iso", "tele", "vice",
    // Verbais
    "entre", "sobre", "sob", "pos", "des",
    // Técnico
    "aero", "astro", "bio", "geo", "foto", "video",
    "eletro", "electro", "mega", "giga", "tera",
    // Outros
    "a", "o", "u",
];

// ═══════════════════════════════════════════════════════
// SUFIXOS (~120)
// ═══════════════════════════════════════════════════════

const SUFFIXES: &[&str] = &[
    // ═══ NOMINAIS ═══
    // Ação / processo
    "ação", "ções", "agem", "agens", "amento", "amentos",
    "imento", "imentos", "ada", "adas", "ida", "idas",
    // Qualidade / estado
    "dade", "dades", "ice", "ices", "eza", "ezas",
    "ismo", "ismos", "ista", "istas", "or", "ores",
    "ência", "ências", "ância", "ânncias",
    // Profissão / agente
    "eiro", "eira", "eiro", "eiras", "ista", "istas",
    "ador", "adora", "adores", "ante", "antes",
    // Coletivo
    "al", "ais", "el", "eis", "ol", "óis",
    "il", "is", "ul", "uis", "ар",
    // Diminutivo
    "inho", "inha", "inhos", "inhas",
    "ito", "ita", "itos", "itas",
    "eta", "etas", "ote", "otes",
    // Aumentativo
    "ão", "ãos", "ona", "onas",
    "arra", "arras", "alhão",
    // Relação
    "eiro", "eira", "ário", "ários",
    "al", "ar", "ial", "ista",
    // ═══ VERBAIS ═══
    // Infinitivo
    "ar", "er", "ir", "or",
    // Gerúndio
    "ando", "endo", "indo", "ondo",
    // Particípio
    "ado", "ados", "ada", "adas",
    "ido", "idos", "ida", "idas",
    // Presente
    "o", "a", "os", "as", "e",
    // Pretérito
    "ei", "aste", "ou", "aram", "eram", "iram",
    // Futuro
    "arei", "aremos", "ará", "arão",
    "erei", "eremos", "erá", "erão",
    "irei", "iremos", "irá", "irão",
    // ═══ ADJETAIVAIS ═══
    "oso", "osa", "osos", "osas",
    "ível", "íveis", "ável", "áveis",
    "al", "ais", "el", "eis", "il", "is",
    "ento", "enta", "entos",
    "mento", "mentos", "nal", "nais",
    "ico", "ica", "icos", "icas",
    "vel", "veis", "il", "is",
    // ═══ ADVERBIAIS ═══
    "mente", "mentes",
    // ═══ GRAU ═══
    "íssimo", "íssima", "íssimos", "íssimas",
    "érrimo", "érrima",
    // ═══ NEGAÇÃO ═══
    "ável", "ível",
];

// ═══════════════════════════════════════════════════════
// RAÍZES (~500) — verbos, substantivos, adjetivos frequentes
// ═══════════════════════════════════════════════════════

const ROOTS: &[&str] = &[
    // ═══ VERBOS AUXILIARES / LINKING ═══
    "ser", "estar", "ter", "haver", "ficar", "andar",
    // ═══ VERBOS DE AÇÃO BÁSICA ═══
    "fazer", "dizer", "ir", "vir", "dar", "ver",
    "saber", "poder", "querer", "achar", "trazer",
    "colocar", "começar", "voltar", "partir", "entrar",
    "sair", "olhar", "correr", "sentir", "passar",
    "ler", "escrever", "comer", "beber", "dormir",
    "abrir", "fechar", "ligar", "desligar", "lig",
    "pagar", "cobrar", "ganhar", "perder", "ganh",
    "perd", "vender", "comprar", "compr", "vend",
    "trabalhar", "trabalh", "estudar", "estud",
    "ensinar", "ensin", "aprender", "aprend",
    "pensar", "pens", "crer", "acreditar", "acredit",
    "conhecer", "conhec", "viver", "viv",
    "morrer", "morr", "nascer", "nasc",
    "cantar", "cant", "dançar", "danç",
    "jogar", "jog", "correr", "cor",
    "andar", "and", "dizer", "diz",
    "falar", "fal", "ouvir", "ouv",
    "ver", "vi", "vir", "v",
    "dar", "d", "ter", "t",
    "ir", "vai", "ser", "s",
    // ═══ SUBSTANTIVOS — Corpo ═══
    "olho", "olhos", "mão", "mãos", "pé", "pés",
    "cabeça", "cabeç", "corpo", "coraç",
    "boca", "nariz", "orelha", "dente",
    "lingua", "sangue", "osso", "pele",
    // ═══ SUBSTANTIVOS — Natureza ═══
    "sol", "lua", "estrela", "céu",
    "mar", "rio", "lago", "montanha",
    "árvore", "floresta", "pedra", "terra",
    "fogo", "ar", "água", "vento",
    "dia", "noite", "manhã", "tarde",
    "ano", "mês", "semana", "hora",
    "tempo", "vez", "momento", "época",
    // ═══ SUBSTANTIVOS — Sociedade ═══
    "homem", "mulher", "criança", "pessoa",
    "gente", "família", "pai", "mãe",
    "filho", "filha", "irmão", "irmã",
    "amigo", "amiga", "vizinho", "vizinha",
    "rei", "rainha", "príncipe", "princesa",
    "presidente", "ministro", "governo",
    "cidade", "país", "mundo", "terra",
    "letra", "palavra", "língua", "texto",
    "livro", "página", "história", "caso",
    "vida", "morte", "amor", "odio",
    "razão", "verdade", "mentira", "justiça",
    "lei", "direito", "dever", "liberdade",
    "trabalho", "emprego", "ofício", "serviço",
    "casa", "morada", "lar", "quarto",
    "rua", "praça", "estrada", "caminho",
    "escola", "aula", "professor", "aluno",
    "igreja", "templo", "altar", "deus",
    "guerra", "paz", "exército", "soldado",
    "dinheiro", "preço", "valor", "moeda",
    "comida", "pão", "carne", "fruta",
    "roupa", "vestido", "sapato", "chapéu",
    "instrumento", "ferramenta", "máquina",
    "animal", "cachorro", "gato", "pássaro",
    "peixe", "inseto", "mosca", "formiga",
    // ═══ ADJETIVOS COMUNS ═══
    "bom", "boa", "mau", "má",
    "grande", "pequeno", "nova", "novo",
    "velho", "jovem", "rico", "pobre",
    "forte", "fraco", "rápido", "lento",
    "quente", "frio", "claro", "escuro",
    "alto", "baixo", "longo", "curto",
    "profundo", "raso", "largo", "estreito",
    "duro", "mole", "suave", "áspero",
    "limpo", "sujo", "bonito", "feio",
    "verdadeiro", "falso", "certo", "errado",
    "feliz", "triste", "alegre", "calmo",
    "novo", "antigo", "primeiro", "último",
    "melhor", "pior", "maior", "menor",
    "próximo", "distante", "perto", "longe",
    // ═══ ADVÉRBIOS ═══
    "não", "sim", "aqui", "ali",
    "onde", "quando", "como", "porque",
    "muito", "pouco", "mais", "menos",
    "sempre", "nunca", "já", "ainda",
    "bem", "mal", "depressa", "devagar",
    "antes", "depois", "agora", "então",
    "também", "também", "apenas", "só",
    "quase", "talvez", "certamente",
    "realmente", "efetivamente", "possivelmente",
    // ═══ PREPOSIÇÕES / CONJUNÇÕES ═══
    "de", "do", "da", "dos", "das",
    "em", "no", "na", "nos", "nas",
    "a", "ao", "à", "aos", "às",
    "por", "para", "com", "sem",
    "sobre", "sob", "entre", "até",
    "depois", "antes", "durante",
    "porque", "porquê", "pois", "mas",
    "e", "ou", "nem", "se",
    "que", "como", "quando", "onde",
    // ═══ PRONOMES ═══
    "eu", "tu", "ele", "ela", "nós",
    "vocês", "eles", "elas", "me",
    "te", "se", "lhe", "nos",
    // ═══ NÚMEROS ═══
    "um", "dois", "três", "quatro",
    "cinco", "seis", "sete", "oito",
    "nove", "dez", "vinte", "trinta",
    "quarenta", "cinquenta", "sessenta",
    "setenta", "oitenta", "noventa",
    "cem", "mil", "milhão", "bilhão",
    // ═══ PALAVRAS DE USO FREQUENTE ═══
    "todo", "toda", "todos", "todas",
    "outro", "outra", "outros", "outras",
    "mesmo", "mesma", "próprio", "própria",
    "certo", "certa", "tal", "tais",
    "cada", "qual", "quais", "quanto",
    "quanta", "quantos", "quantas",
    "que", "qual", "quem", "onde",
    "este", "esta", "estes", "estas",
    "esse", "essa", "esses", "essas",
    "aquele", "aquela", "aqueles", "aquelas",
    "isto", "isso", "aquilo",
];

pub struct AffixVocabulary {
    pub prefix_to_id: HashMap<String, u32>,
    pub suffix_to_id: HashMap<String, u32>,
    pub root_to_id: HashMap<String, u32>,
    pub id_to_affix: HashMap<u32, String>,
    pub affix_bits: HashMap<u32, u32>,
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

        for prefix in PREFIXES {
            v.register_affix(prefix, "prefix");
        }
        for suffix in SUFFIXES {
            v.register_affix(suffix, "suffix");
        }
        for root in ROOTS {
            v.register_affix(root, "root");
        }

        v.generate_affix_bits();
        v
    }

    fn register_affix(&mut self, affix: &str, kind: &str) {
        if self.prefix_to_id.contains_key(affix)
            || self.suffix_to_id.contains_key(affix)
            || self.root_to_id.contains_key(affix)
        {
            return;
        }

        let id = self.next_id;
        self.next_id += 1;
        self.id_to_affix.insert(id, affix.to_string());

        match kind {
            "prefix" => { self.prefix_to_id.insert(affix.to_string(), id); }
            "suffix" => { self.suffix_to_id.insert(affix.to_string(), id); }
            "root" => { self.root_to_id.insert(affix.to_string(), id); }
            _ => {}
        }
    }

    fn generate_affix_bits(&mut self) {
        for (&id, affix) in &self.id_to_affix.clone() {
            let bits = self.affix_hash(affix);
            self.affix_bits.insert(id, bits);
        }
    }

    /// Hash determinístico: cada afixo → 32 bits únicos
    /// Bits codificam: comprimento, vogais, consoantes, padrão
    fn affix_hash(&self, affix: &str) -> u32 {
        let mut h: u32 = 0x811c9dc5;
        for byte in affix.bytes() {
            h ^= byte as u32;
            h = h.wrapping_mul(0x01000193);
        }

        let chars: Vec<char> = affix.chars().collect();
        let len = chars.len() as u32;

        // Bits 0-5: comprimento (0-63)
        let mut bits = len & 0x3F;

        // Bits 6-13: primeira letra (a-z)
        if let Some(&first) = chars.first() {
            if first.is_ascii_lowercase() {
                bits |= ((first as u32 - 'a' as u32) & 0x1F) << 6;
            }
        }

        // Bits 14-21: última letra
        if let Some(&last) = chars.last() {
            if last.is_ascii_lowercase() {
                bits |= ((last as u32 - 'a' as u32) & 0x1F) << 14;
            }
        }

        // Bits 22-25: número de vogais
        let vowels = chars.iter().filter(|c| "aeiou".contains(**c)).count() as u32;
        bits |= (vowels & 0x0F) << 22;

        // Bits 26-29: número de consoantes
        let consonants = chars.iter().filter(|c| c.is_ascii_lowercase() && !"aeiou".contains(**c)).count() as u32;
        bits |= (consonants & 0x0F) << 26;

        // Bits 30-31: tipo (00=prefix, 01=suffix, 10=root)
        // Será preenchido depois se necessário

        bits ^ h
    }

    /// Decompor palavra em (prefix_id, root_id, suffix_id)
    /// Algoritmo guloso: maior prefixo + maior sufixo = raiz
    pub fn decompose(&self, word: &str) -> (Option<u32>, Option<u32>, Option<u32>) {
        let word_lower = word.to_lowercase();
        let chars: Vec<char> = word_lower.chars().collect();
        let len = chars.len();

        if len < 2 {
            return (None, None, None);
        }

        // 1. Encontrar maior prefixo
        let mut best_prefix: Option<u32> = None;
        let mut best_plen = 0;
        for (prefix, &id) in &self.prefix_to_id {
            let pchars: Vec<char> = prefix.chars().collect();
            let plen = pchars.len();
            if plen < len && plen > best_plen {
                if chars[..plen] == pchars[..] {
                    best_prefix = Some(id);
                    best_plen = plen;
                }
            }
        }

        // 2. Encontrar maior sufixo (sem sobrepor com prefixo)
        let mut best_suffix: Option<u32> = None;
        let mut best_slen = 0;
        for (suffix, &id) in &self.suffix_to_id {
            let schars: Vec<char> = suffix.chars().collect();
            let slen = schars.len();
            if slen < len && best_plen + slen < len && slen > best_slen {
                if chars[len - slen..] == schars[..] {
                    best_suffix = Some(id);
                    best_slen = slen;
                }
            }
        }

        // 3. Raiz = o que sobra
        let root_len = len - best_plen - best_slen;
        let best_root = if root_len > 0 {
            let root_str: String = chars[best_plen..best_plen + root_len].iter().collect();
            self.root_to_id.get(&root_str).copied()
        } else {
            None
        };

        (best_prefix, best_root, best_suffix)
    }

    /// Embedding de palavra = XOR dos bits dos seus afixos
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

        // Fallback: hash da palavra inteira
        if embedding == 0 {
            let mut h: u32 = 0x811c9dc5;
            for byte in word.bytes() {
                h ^= byte as u32;
                h = h.wrapping_mul(0x01000193);
            }
            embedding = h;
        }

        embedding
    }

    /// Similaridade morfológica: quantos afixos compartilham?
    pub fn morphological_similarity(&self, word1: &str, word2: &str) -> f32 {
        let (p1, r1, s1) = self.decompose(word1);
        let (p2, r2, s2) = self.decompose(word2);

        let mut score = 0.0f32;
        let mut total = 0.0f32;

        // Prefixo: 0.25
        total += 0.25;
        if p1.is_some() && p1 == p2 { score += 0.25; }

        // Raiz: 0.50 (mais importante)
        total += 0.50;
        if r1.is_some() && r1 == r2 { score += 0.50; }

        // Sufixo: 0.25
        total += 0.25;
        if s1.is_some() && s1 == s2 { score += 0.25; }

        score / total
    }

    /// Encontrar k palavras mais similares morfologicamente
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

    /// Estatísticas
    pub fn stats(&self) -> AffixStats {
        AffixStats {
            n_prefixes: self.prefix_to_id.len(),
            n_suffixes: self.suffix_to_id.len(),
            n_roots: self.root_to_id.len(),
            n_total: self.next_id as usize - 1,
        }
    }

    /// Decompor e mostrar como string legível
    pub fn decompose_str(&self, word: &str) -> String {
        let (p, r, s) = self.decompose(word);
        let p_str = p.map(|id| self.id_to_affix[&id].as_str()).unwrap_or("-");
        let r_str = r.map(|id| self.id_to_affix[&id].as_str()).unwrap_or("-");
        let s_str = s.map(|id| self.id_to_affix[&id].as_str()).unwrap_or("-");
        format!("{} = {}+{}+{}", word, p_str, r_str, s_str)
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
            "Vocabulário morfológico: {} prefixos + {} sufixos + {} raízes = {} componentes",
            self.n_prefixes, self.n_suffixes, self.n_roots, self.n_total
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decompose() {
        let v = AffixVocabulary::new();
        println!("{}", v.decompose_str("invisível"));
        println!("{}", v.decompose_str("desconhecido"));
        println!("{}", v.decompose_str("bonitas"));
        println!("{}", v.decompose_str("correndo"));
        println!("{}", v.decompose_str("brasileiro"));
        println!("{}", v.decompose_str("anticonstitucionalmente"));
        println!("{}", v.decompose_str("microcomputador"));
        println!("{}", v.decompose_str("desenvolvimento"));
        println!("{}", v.decompose_str("internacionalização"));
    }

    #[test]
    fn test_similarity() {
        let v = AffixVocabulary::new();

        let pairs = vec![
            ("casa", "casas"),
            ("bonito", "bonita"),
            ("invisível", "visível"),
            ("correr", "correndo"),
            ("desconhecido", "conhecido"),
            ("casa", "carro"),
        ];

        for (w1, w2) in &pairs {
            let sim = v.morphological_similarity(w1, w2);
            println!("{} vs {}: {:.2}", w1, w2, sim);
        }
    }
}
