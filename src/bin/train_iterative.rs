use amadeus::amadeus_m::triple_grammar::TripleGrammar;
use amadeus::amadeus_m::token7::Token7;
use amadeus::amadeus_m::hmm::{HmmPosTagger, TAG_SUBST, TAG_VERBO, TAG_ADJ, TAG_ART, TAG_ADV, TAG_PREP, TAG_CONJ, TAG_PONT};
use amadeus::amadeus_m::morph_vocab::AffixVocabulary;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

struct SimpleCompiler {
    lexicon: HashMap<String, (u32, u16, u8, u16)>,
    roots: Vec<String>,
    next_id: u32,
    hmm: HmmPosTagger,
}

impl SimpleCompiler {
    fn new() -> Self {
        let mut c = Self {
            lexicon: HashMap::new(),
            roots: vec![String::new()],
            next_id: 1,
            hmm: HmmPosTagger::new(),
        };
        for p in &[",", ".", "!", "?", ";", ":", "—", "-", "\"", "(", ")", "..."] {
            let morph = 6u16;
            c.lexicon.insert(p.to_string(), (0, morph, 6, 0));
        }
        c
    }

    fn tokenize_sentences(text: &str) -> Vec<Vec<String>> {
        text.split(|c| c == '.' || c == '!' || c == '?')
            .filter(|s| s.trim().len() > 10)
            .map(|sent| {
                sent.split(|c: char| c.is_whitespace() || c == ',' || c == ';' || c == ':' || c == '(' || c == ')')
                    .filter(|w| !w.trim().is_empty())
                    .map(|w| w.to_lowercase())
                    .collect::<Vec<_>>()
            })
            .filter(|words| words.len() >= 2)
            .collect()
    }

    fn train_hmm(&mut self, text: &str) {
        let sentences = Self::tokenize_sentences(text);
        self.hmm.train(&sentences);
    }

    fn compile(&mut self, text: &str) -> Vec<Token7> {
        let mut tokens = Vec::new();
        let words: Vec<String> = text.split(|c: char| c.is_whitespace() || c == ',' || c == '.' || c == '!' || c == '?' || c == ';' || c == ':')
            .filter(|w| !w.trim().is_empty())
            .map(|w| w.to_lowercase())
            .collect();
        if words.is_empty() { return tokens; }
        let tags = self.hmm.viterbi(&words);
        for (i, w) in words.iter().enumerate() {
            if let Some(&(id, morph, _class, style)) = self.lexicon.get(w) {
                tokens.push(Token7::new(id, morph).with_style(style));
                continue;
            }
            let tag = tags[i];
            let (id, morph, _class, style) = self.register_word(w, tag);
            tokens.push(Token7::new(id, morph).with_style(style));
        }
        tokens
    }

    fn register_word(&mut self, word: &str, tag: usize) -> (u32, u16, u8, u16) {
        let id = self.next_id;
        self.next_id += 1;
        if id as usize >= self.roots.len() {
            self.roots.resize(id as usize + 1, String::new());
        }
        self.roots[id as usize] = word.to_string();
        let (class, morph_bits) = match tag {
            TAG_SUBST => (0, (id as u16).wrapping_mul(0x9E37) & 0x00FF),
            TAG_VERBO => (1, 0b0000_1000u16),
            TAG_ADJ => (2, 0b0000_0100u16),
            TAG_ART => (3, 0b0100_0000u16),
            TAG_ADV => (4, 0b0000_0010u16),
            TAG_PREP => (5, 0b0000_0011u16),
            TAG_CONJ => (5, 0b0000_1100u16),
            TAG_PONT => (6, 0u16),
            _ => (0, (id as u16).wrapping_mul(0x9E37) & 0x00FF),
        };
        let style = word_style(id, class);
        self.lexicon.insert(word.to_string(), (id, morph_bits | (class as u16), class, style));
        (id, morph_bits | (class as u16), class, style)
    }
}

fn word_style(id: u32, class: u8) -> u16 {
    let hash = (id as u16).wrapping_mul(0x9E37).wrapping_add(class as u16 * 0x7C5);
    hash & 0x3F
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  AMADEUS — TREINO MULTI-ITERAÇÃO + TUNING              ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();

    // ─── Carregar corpus ───
    println!("▸ Carregando corpus...");
    let wiki_dir = "training/wikipedia";
    let narrative_dir = "training/narrative";
    let mut all_text = String::new();

    if Path::new(wiki_dir).exists() {
        for entry in fs::read_dir(wiki_dir).unwrap().flatten() {
            let path = entry.path();
            if path.extension().map(|e| e == "txt").unwrap_or(false) {
                if let Ok(content) = fs::read_to_string(&path) {
                    if content.len() > 100 {
                        all_text.push_str(&content);
                        all_text.push('\n');
                    }
                }
            }
        }
    }
    if Path::new(narrative_dir).exists() {
        for entry in fs::read_dir(narrative_dir).unwrap().flatten() {
            let path = entry.path();
            if path.extension().map(|e| e == "txt").unwrap_or(false) {
                if let Ok(content) = fs::read_to_string(&path) {
                    if content.len() > 1000 {
                        all_text.push_str(&content);
                        all_text.push('\n');
                    }
                }
            }
        }
    }
    println!("  {} KB totais", all_text.len() / 1024);

    // ─── Compilar tokens ───
    println!("▸ Compilando tokens...");
    let mut compiler = SimpleCompiler::new();
    compiler.train_hmm(&all_text);

    let sentences: Vec<&str> = all_text.split(|c| c == '.' || c == '!' || c == '?')
        .filter(|s| s.trim().len() > 10)
        .collect();

    let mut all_tokens: Vec<Token7> = Vec::new();
    for sentence in &sentences {
        let tokens = compiler.compile(sentence);
        if tokens.len() >= 2 {
            all_tokens.extend_from_slice(&tokens);
        }
    }
    println!("  {} tokens compilados", all_tokens.len());

    // ─── Vocabulário morfológico ───
    let morph_vocab = AffixVocabulary::new();
    println!("  {}", morph_vocab.stats());

    // ─── Multi-iteração: 5 rodadas de treino completo ───
    let n_iterations = 5;
    let configs = vec![
        ("temp=0.8, alpha=0.3", 0.8, 0.3),
        ("temp=0.8, alpha=0.5", 0.8, 0.5),
        ("temp=1.0, alpha=0.4", 1.0, 0.4),
        ("temp=0.6, alpha=0.6", 0.6, 0.6),
        ("temp=1.2, alpha=0.4", 1.2, 0.4),
    ];

    let seeds = vec!["o Brasil", "a música", "a ciência", "o futebol"];

    for iter in 0..n_iterations {
        println!();
        println!("═══════════════════════════════════════════════════════════");
        println!("  ITERAÇÃO {}/{}", iter + 1, n_iterations);
        println!("═══════════════════════════════════════════════════════════");

        // Treinar gramática
        let mut grammar = TripleGrammar::new(3);

        let mut processed = 0usize;
        for sentence in &sentences {
            let tokens = compiler.compile(sentence);
            if tokens.len() < 2 { continue; }
            let tokens_with_deps = amadeus::amadeus_m::syntax::assign_dependencies(&tokens);
            grammar.train(&tokens_with_deps);
            processed += tokens_with_deps.len();
        }
        println!("  Gramática treinada: {} tokens", processed);

        // GRAPH morfológico
        let mut graph: HashMap<u32, u32> = HashMap::new();
        for (word, &(lex_id, _, _, _)) in &compiler.lexicon {
            if lex_id == 0 { continue; }
            let embedding = morph_vocab.word_embedding(word);
            graph.insert(lex_id, embedding);
        }
        grammar.graph = graph;
        grammar.graph_alpha = 0.3;

        // Treinar SNN
        grammar.build_snn();
        grammar.train_snn_iterative(10, 0.001);

        // Construir reverse map (precisa ser após o treino do compiler)
        // Mas compiler.lexicon é borrowado pelo loop...
        // Solução: coletar IDs únicos antes

        // Testar cada config
        for (name, temp, alpha) in &configs {
            grammar.temperature = *temp;
            grammar.cohesion_alpha = *alpha;

            println!();
            println!("  Config: {} (temp={}, alpha={})", name, temp, alpha);

            for seed in &seeds {
                let seed_tokens = compiler.compile(seed);
                if seed_tokens.is_empty() { continue; }

                let generated = grammar.generate_text(seed_tokens[0].lex, 2);

                // Decompilar inline
                let words: Vec<String> = generated.iter()
                    .filter(|t| t.lex != 0)
                    .filter_map(|t| {
                        compiler.lexicon.iter()
                            .find(|(_, v)| v.0 == t.lex)
                            .map(|(w, _)| w.clone())
                    })
                    .collect();

                let word_refs: Vec<&str> = words.iter().map(|s| s.as_str()).collect();
                println!("    \"{}\" → {}", seed, word_refs.join(" "));
            }
        }
    }

    println!();
    println!("═══════════════════════════════════════════════════════════");
    println!("  TREINO MULTI-ITERAÇÃO COMPLETO");
    println!("═══════════════════════════════════════════════════════════");
}
