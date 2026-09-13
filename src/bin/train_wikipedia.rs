use amadeus::amadeus_m::triple_grammar::TripleGrammar;
use amadeus::amadeus_m::token7::Token7;
use amadeus::amadeus_m::hmm::{HmmPosTagger, TAG_SUBST, TAG_VERBO, TAG_ADJ, TAG_ART, TAG_ADV, TAG_PREP, TAG_CONJ, TAG_PONT};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Compilador simplificado que aprende vocabulário diretamente do texto
/// Usa HMM para classificação automática de palavras
struct SimpleCompiler {
    lexicon: HashMap<String, (u32, u16, u8, u16)>, // word -> (id, morph, class, style)
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

        // Adicionar pontuação (class 6 = PONTUACAO)
        for p in &[",", ".", "!", "?", ";", ":", "—", "-", "\"", "(", ")", "..."] {
            let morph = 6u16; // PONTUACAO
            c.lexicon.insert(p.to_string(), (0, morph, 6, 0));
        }

        c
    }

    /// Tokenizar texto bruto em sentenças de palavras
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

    /// Treinar o HMM com texto bruto
    fn train_hmm(&mut self, text: &str) {
        println!("  Tokenizando para HMM...");
        let sentences = Self::tokenize_sentences(text);
        println!("  {} sentenças para treinar HMM", sentences.len());

        self.hmm.train(&sentences);

        let stats = self.hmm.stats();
        println!("  HMM treinado:");
        let tag_names = ["SUBST", "VERBO", "ADJ", "ART", "ADV", "PREP", "CONJ", "PONT"];
        for (i, name) in tag_names.iter().enumerate() {
            println!("    {}: {:.1}% prior, {} palavras",
                name, stats.prior[i] * 100.0, stats.word_counts[i]);
        }

        // Mostrar top 3 palavras por tag
        let top = self.hmm.top_words_per_tag(3);
        for (i, name) in tag_names.iter().enumerate() {
            let words: Vec<&str> = top[i].iter().map(|(w, _)| w.as_str()).collect();
            if !words.is_empty() {
                println!("    {} top: {}", name, words.join(", "));
            }
        }
    }

    /// Compilar texto usando HMM para classificação
    fn compile(&mut self, text: &str) -> Vec<Token7> {
        let mut tokens = Vec::new();
        let words: Vec<String> = text.split(|c: char| c.is_whitespace() || c == ',' || c == '.' || c == '!' || c == '?' || c == ';' || c == ':')
            .filter(|w| !w.trim().is_empty())
            .map(|w| w.to_lowercase())
            .collect();

        if words.is_empty() { return tokens; }

        // Usar HMM para taggear a frase inteira
        let tags = self.hmm.viterbi(&words);

        for (i, w) in words.iter().enumerate() {
            // Verificar se já está no léxico
            if let Some(&(id, morph, _class, style)) = self.lexicon.get(w) {
                tokens.push(Token7::new(id, morph).with_style(style));
                continue;
            }

            // HMM classificou esta palavra
            let tag = tags[i];
            let (id, morph, class, style) = self.register_word(w, tag);
            tokens.push(Token7::new(id, morph).with_style(style));
        }

        tokens
    }

    /// Registrar palavra no léxico com a classe dada pelo HMM
    fn register_word(&mut self, word: &str, tag: usize) -> (u32, u16, u8, u16) {
        let id = self.next_id;
        self.next_id += 1;

        if id as usize >= self.roots.len() {
            self.roots.resize(id as usize + 1, String::new());
        }
        self.roots[id as usize] = word.to_string();

        // Mapear tag HMM para classe Amadeus
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
    println!("║  AMADEUS — TREINAMENTO MISTO (WIKIPÉDIA + NARRATIVO)  ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();

    // ─── FASE 1: Coletar textos ───
    println!("▸ FASE 1: Coletando textos...");

    let wiki_dir = "training/wikipedia";
    let narrative_dir = "training/narrative";
    let mut all_text = String::new();
    let mut file_count = 0;
    let mut wiki_chars = 0usize;
    let mut narrative_chars = 0usize;

    // Carregar Wikipédia
    if Path::new(wiki_dir).exists() {
        println!("  📚 Wikipédia:");
        for entry in fs::read_dir(wiki_dir).unwrap().flatten() {
            let path = entry.path();
            if path.extension().map(|e| e == "txt").unwrap_or(false) {
                if let Ok(content) = fs::read_to_string(&path) {
                    let file_name = path.file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("?");
                    if content.len() > 100 {
                        println!("    → {} ({} KB)", file_name, content.len() / 1024);
                        all_text.push_str(&content);
                        all_text.push('\n');
                        wiki_chars += content.len();
                        file_count += 1;
                    }
                }
            }
        }
    }

    // Carregar narrativos
    if Path::new(narrative_dir).exists() {
        println!("  📖 Narrativos (romances):");
        for entry in fs::read_dir(narrative_dir).unwrap().flatten() {
            let path = entry.path();
            if path.extension().map(|e| e == "txt").unwrap_or(false) {
                if let Ok(content) = fs::read_to_string(&path) {
                    let file_name = path.file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("?");
                    if content.len() > 1000 {
                        println!("    → {} ({} KB)", file_name, content.len() / 1024);
                        all_text.push_str(&content);
                        all_text.push('\n');
                        narrative_chars += content.len();
                        file_count += 1;
                    }
                }
            }
        }
    }

    println!("  {} arquivos totais", file_count);
    println!("  Wikipédia: {} KB | Narrativos: {} KB | Total: {} KB",
        wiki_chars / 1024, narrative_chars / 1024, all_text.len() / 1024);
    println!();

    // ─── FASE 2: Treinar HMM ───
    println!("▸ FASE 2: Treinando HMM para classificação automática...");

    let mut compiler = SimpleCompiler::new();
    compiler.train_hmm(&all_text);
    println!();

    // ─── FASE 3: Compilar tokens com HMM ───
    println!("▸ FASE 3: Compilando tokens com HMM...");

    // Dividir em frases
    let sentences: Vec<&str> = all_text
        .split(|c| c == '.' || c == '!' || c == '?')
        .filter(|s| s.trim().len() > 10)
        .collect();

    println!("  {} frases extraídas", sentences.len());

    // Compilar cada frase
    let mut all_tokens: Vec<Token7> = Vec::new();
    for sentence in &sentences {
        let tokens = compiler.compile(sentence);
        if tokens.len() >= 2 {
            all_tokens.extend_from_slice(&tokens);
        }
    }

    println!("  {} tokens compilados", all_tokens.len());
    println!();

    // ─── FASE 4: Treinar gramática ───
    println!("▸ FASE 4: Treinando gramática...");

    let mut grammar = TripleGrammar::new(3);

    // Treinar por frase (não em todos os tokens de uma vez - O(n²))
    let mut processed = 0usize;
    for sentence in &sentences {
        let tokens = compiler.compile(sentence);
        if tokens.len() < 2 { continue; }
        let tokens_with_deps = amadeus::amadeus_m::syntax::assign_dependencies(&tokens);
        grammar.train(&tokens_with_deps);
        processed += tokens_with_deps.len();
        if processed % 10000 < 100 {
            println!("  ... {} tokens processados", processed);
        }
    }

    println!("  Treinamento concluído");
    println!();

    // ─── FASE 5: Construir GRAPH embeddings ───
    println!("▸ FASE 5: Construindo GRAPH embeddings...");

    // Usar random indexing para criar embeddings de 32 bits
    // Cada palavra recebe um vetor esparso baseado em coocorrência
    let mut graph: HashMap<u32, u32> = HashMap::new();
    let mut rng = fastrand::Rng::with_seed(42);

    // Criar embeddings baseados em coocorrência de janela
    let window_size = 3;
    let mut cooccurrence: HashMap<(u32, u32), f32> = HashMap::new();

    for window in all_tokens.windows(window_size) {
        let center = window[window_size / 2];
        if center.lex == 0 { continue; }

        for i in 0..window.len() {
            if i == window_size / 2 { continue; }
            let context = window[i];
            if context.lex == 0 { continue; }

            let key = if center.lex < context.lex {
                (center.lex, context.lex)
            } else {
                (context.lex, center.lex)
            };
            *cooccurrence.entry(key).or_insert(0.0) += 1.0;
        }
    }

    // Criar embeddings usando random projections
    let n_dims = 32;
    let mut projection: Vec<Vec<i8>> = Vec::new();
    for _ in 0..n_dims {
        let mut row: Vec<i8> = Vec::new();
        for _ in 0..compiler.roots.len() {
            row.push(rng.i8(-1..=1));
        }
        projection.push(row);
    }

    // Calcular embedding para cada lex_id
    let mut lex_ids: Vec<u32> = compiler.lexicon.values()
        .map(|&(id, _, _, _)| id)
        .filter(|&id| id > 0)
        .collect();
    lex_ids.sort();
    lex_ids.dedup();

    // Inverter co-ocorrências: lex_id → [(outro, count)]
    let mut lex_cooccur: HashMap<u32, Vec<(u32, f32)>> = HashMap::new();
    for (&(a, b), &count) in &cooccurrence {
        lex_cooccur.entry(a).or_default().push((b, count));
        lex_cooccur.entry(b).or_default().push((a, count));
    }

    for &lex_id in &lex_ids {
        let mut embedding: u32 = 0;

        if let Some(neighbors) = lex_cooccur.get(&lex_id) {
            for &(other, _count) in neighbors {
                if (other as usize) < projection[0].len() {
                    for dim in 0..n_dims {
                        if projection[dim][other as usize] > 0 {
                            embedding ^= 1 << dim;
                        }
                    }
                }
            }
        }

        graph.insert(lex_id, embedding);
    }

    // Copiar GRAPH para a gramática
    grammar.graph = graph.clone();
    grammar.graph_alpha = 0.3; // Peso da modulação semântica

    println!("  GRAPH: {} embeddings criados", graph.len());
    println!();

    // ─── FASE 6: Métricas ───
    println!("▸ FASE 6: Métricas do treinamento...");

    let lex_count = grammar.lexicon.len();
    let t2_count = grammar.agreement.len();
    let t3_count = grammar.lexicon.len();
    let graph_count = grammar.graph.len();

    println!("  Léxico: {} entradas", lex_count);
    println!("  T2 (concordância): {} padrões", t2_count);
    println!("  T3 (seleção): {} entradas", t3_count);
    println!("  GRAPH: {} embeddings", graph_count);

    // Distribuição de classes
    let mut class_counts = [0u32; 8];
    let lex_size = compiler.lexicon.len();
    for &(_, _, class, _) in compiler.lexicon.values() {
        if (class as usize) < class_counts.len() {
            class_counts[class as usize] += 1;
        }
    }
    let class_names = ["SUBST", "VERBO", "ADJ", "ART/PRON", "ADV", "PREP/CONJ", "PONT", "OUTRO"];
    println!("\n  Distribuição de classes ({} entradas no léxico):", lex_size);
    for (i, name) in class_names.iter().enumerate() {
        let pct = class_counts[i] as f32 / lex_size as f32 * 100.0;
        println!("    {}: {} ({:.1}%)", name, class_counts[i], pct);
    }
    println!();

    // ─── FASE 7: Gerar texto com GRAPH modulation ───
    println!("▸ FASE 7: Gerando texto com GRAPH modulation...");

    // Sementes variadas
    let seeds = vec![
        "o Brasil",
        "a música",
        "a ciência",
        "o futebol",
        "a história",
        "a natureza",
        "o cinema",
        "a arte",
        "a vida",
    ];

    for seed in &seeds {
        let seed_tokens = compiler.compile(seed);
        if seed_tokens.is_empty() { continue; }

        // Gerar com GRAPH modulation
        let generated = grammar.generate(&seed_tokens, 15);

        // Decompilar
        let reverse: HashMap<u32, &str> = compiler.lexicon.iter()
            .map(|(w, &(id, _, _, _))| (id, w.as_str()))
            .collect();

        let words: Vec<&str> = generated.iter()
            .filter(|t| t.lex != 0)
            .filter_map(|t| reverse.get(&t.lex).copied())
            .collect();

        println!("  \"{}\" → {}", seed, words.join(" "));
    }

    println!();

    // ─── FASE 8: Salvar ───
    println!("▸ FASE 8: Salvando gramática treinada...");

    let bin_path = "training/wikipedia_grammar.bin";
    let gguf_path = "training/wikipedia_grammar.gguf";

    grammar.save(bin_path).unwrap();
    println!("  BIN: {} bytes", fs::metadata(bin_path).unwrap().len());

    grammar.save_gguf(gguf_path, &compiler.lexicon, &compiler.roots).unwrap();
    println!("  GGUF: {} bytes", fs::metadata(gguf_path).unwrap().len());

    println!();

    // ─── FASE 9: Treinar SNN com dados reais ───
    println!("▸ FASE 9: Treinando SNN com dados reais...");

    // Construir SNN a partir da gramática treinada
    grammar.build_snn();

    println!("  SNN construída a partir da gramática");
    if let Some(ref snn) = grammar.snn {
        println!("  Pesos SNN: {} synapses IH, {} synapses HO",
            snn.synapses_ih.len(), snn.synapses_ho.len());
    }

    // ─── FASE 10: STDP iterativo ───
    println!();
    println!("▸ FASE 10: Treinamento STDP iterativo...");

    grammar.train_snn_iterative(10, 0.01);

    println!();
    println!("═══════════════════════════════════════════════════════════");
    println!("  TREINAMENTO MISTO COMPLETO");
    println!("═══════════════════════════════════════════════════════════");
}
