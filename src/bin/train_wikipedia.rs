use amadeus::amadeus_m::triple_grammar::TripleGrammar;
use amadeus::amadeus_m::token7::Token7;
use amadeus::amadeus_m::hmm::{HmmPosTagger, TAG_SUBST, TAG_VERBO, TAG_ADJ, TAG_ART, TAG_ADV, TAG_PREP, TAG_CONJ, TAG_PONT};
use amadeus::amadeus_m::morph_vocab::AffixVocabulary;
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

        let style = classify_style(word, class);
        self.lexicon.insert(word.to_string(), (id, morph_bits | (class as u16), class, style));
        (id, morph_bits | (class as u16), class, style)
    }
}

/// Classificar estilo de uma palavra: 0=neutro, 1=formal, 2=informal
/// Baseado em: comprimento, padrões, sufixos, domínio
fn classify_style(word: &str, class: u8) -> u16 {
    let w = word.to_lowercase();
    let len = w.len();

    // ─── INFORMAL (2) ───
    // Abreviações coloquiais
    if w == "vc" || w == "voc" || w == "tb" || w == "tbm" || w == "msg"
        || w == "qdo" || w == "pq" || w == "ai" || w == "ah" || w == "oh"
        || w == "né" || w == "né" || w == "tô" || w == "tá" || w == "flw"
        || w == "blz" || w == "vd" || w == "q" || w == "n" || w == "s"
        || w == "pf" || w == "obg" || w == "vlw" || w == "sassudo"
        || w == "mano" || w == "cara" || w == "tipo" || w == "bicho"
        || w == "caraca" || w == "eita" || w == "oxe" || w == "uai"
    {
        return 2;
    }

    // Diminutivos coloquiais (informal)
    if w.ends_with("inho") || w.ends_with("inha") || w.ends_with("ito") || w.ends_with("ita") {
        if len <= 8 {
            return 2;
        }
    }

    // Gírias / vocabulary coloquial
    if w == "trampo" || w == "moleque" || w == "mina" || w == "zica"
        || w == "balada" || w == "rolê" || w == "suave" || w == "massa"
        || w == "da hora" || w == "top" || w == "irado" || w == "sinistro"
    {
        return 2;
    }

    // ─── FORMAL (1) ───
    // Sufixos formais / técnicos
    if w.ends_with("ão") && (class == 0 || class == 2) && len > 8 {
        return 1; // substantivos/adjetivos longos em -ão tendem a ser formais
    }

    // Termos técnicos/científicos
    if w.contains("olog") || w.contains("ismo") && len > 8
        || w.ends_with("idade") && len > 6
        || w.ends_with("ário") || w.ends_with("eiro") && len > 7
        || w.ends_with("ância") || w.ends_with("ência")
        || w.ends_with("ível") || w.ends_with("ável")
    {
        return 1;
    }

    // Palavras muito longas tendem a ser formais
    if len > 12 {
        return 1;
    }

    // Termos de domínio formal (jurídico, médico, acadêmico)
    if w.starts_with("art") || w.starts_with("parágrafo") || w.starts_with("alínea")
        || w.starts_with("inciso") || w.starts_with("cláusula")
        || w.starts_with("diagnóstico") || w.starts_with("prognóstico")
        || w.starts_with("tratamento") || w.starts_with("prescrição")
        || w.starts_with("norma") || w.starts_with("regulament")
    {
        return 1;
    }

    // ─── NEUTRO (0) ───
    // Maioria das palavras
    0
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

    // ─── FASE 5: Construir GRAPH embeddings via SVD ───
    println!("▸ FASE 5: Construindo GRAPH embeddings (SVD sobre co-ocorrência)...");

    // Criar vocabulário de afixos
    let morph_vocab = AffixVocabulary::new();
    let stats = morph_vocab.stats();
    println!("  {}", stats);

    // Coletar todas as palavras do léxico
    let word_list: Vec<String> = compiler.lexicon.keys().cloned().collect();

    // ─── 1. Construir matriz de co-ocorrência ───
    let window_size = 5;
    let mut cooccurrence: HashMap<(u32, u32), f32> = HashMap::new();
    let mut word_freq: HashMap<u32, f32> = HashMap::new();
    let mut total_pairs = 0.0f32;

    for window in all_tokens.windows(window_size) {
        let center = window[window_size / 2];
        if center.lex == 0 { continue; }
        *word_freq.entry(center.lex).or_insert(0.0) += 1.0;

        for i in 0..window.len() {
            if i == window_size / 2 { continue; }
            let context = window[i];
            if context.lex == 0 { continue; }

            let dist = (i as i32 - window_size as i32 / 2).unsigned_abs() as f32;
            let weight = 1.0 / (1.0 + dist);

            let key = if center.lex < context.lex {
                (center.lex, context.lex)
            } else {
                (context.lex, center.lex)
            };
            *cooccurrence.entry(key).or_insert(0.0) += weight;
            total_pairs += weight;
        }
    }

    // ─── 2. PPMI sobre co-ocorrência ───
    let mut pmi_scores: HashMap<(u32, u32), f32> = HashMap::new();
    for (&(a, b), &count) in &cooccurrence {
        let p_ab = count / total_pairs;
        let p_a = word_freq.get(&a).copied().unwrap_or(1.0) / total_pairs;
        let p_b = word_freq.get(&b).copied().unwrap_or(1.0) / total_pairs;
        let pmi = (p_ab / (p_a * p_b)).max(1e-10).log2();
        if pmi > 0.0 {
            pmi_scores.insert((a, b), pmi);
        }
    }

    // ─── 2.1 Normalizar PPMI por linha (frequência da palavra) ───
    // Isso evita que palavras frequentes dominem o SVD
    let mut row_norms: HashMap<u32, f32> = HashMap::new();
    for (&(a, _), &ppmi) in &pmi_scores {
        *row_norms.entry(a).or_insert(0.0) += ppmi * ppmi;
    }
    for norm in row_norms.values_mut() {
        *norm = norm.sqrt().max(1e-10);
    }

    // PPMI normalizado: ppmi / ||row||
    let mut ppmi_normalized: HashMap<(u32, u32), f32> = HashMap::new();
    for (&(a, b), &ppmi) in &pmi_scores {
        if let Some(&row_norm) = row_norms.get(&a) {
            ppmi_normalized.insert((a, b), ppmi / row_norm);
        }
    }

    // Usar PPMI normalizado para SVD
    let pmi_scores_for_svd = &ppmi_normalized;

    // ─── 3. SVD simplificado: power iteration para top-k vetores ───
    // Mapear lex_ids para índices — apenas top-2000 mais frequentes
    let mut freq_sorted: Vec<(u32, f32)> = word_freq.iter()
        .map(|(&id, &freq)| (id, freq))
        .collect();
    freq_sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    freq_sorted.truncate(2000);

    let n_words = freq_sorted.len();
    let lex_to_idx: HashMap<u32, usize> = freq_sorted.iter()
        .enumerate()
        .map(|(i, &(id, _))| (id, i))
        .collect();

    let n_dims = 32;
    let n_iters = 5;

    let mut embeddings: Vec<Vec<f32>> = vec![vec![0.0; n_dims]; n_words];

    let mut rng = fastrand::Rng::with_seed(42);
    for dim in 0..n_dims {
        let mut v: Vec<f32> = (0..n_words).map(|_| rng.f32() - 0.5).collect();
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 { v.iter_mut().for_each(|x| *x /= norm); }

        for _ in 0..n_iters {
            let mut u: Vec<f32> = vec![0.0; n_words];
            for (&(a, b), &ppmi) in pmi_scores_for_svd {
                if let (Some(&ia), Some(&ib)) = (lex_to_idx.get(&a), lex_to_idx.get(&b)) {
                    u[ia] += ppmi * v[ib];
                    u[ib] += ppmi * v[ia];
                }
            }

            let mut v_new: Vec<f32> = vec![0.0; n_words];
            for (&(a, b), &ppmi) in pmi_scores_for_svd {
                if let (Some(&ia), Some(&ib)) = (lex_to_idx.get(&a), lex_to_idx.get(&b)) {
                    v_new[ia] += ppmi * u[ib];
                    v_new[ib] += ppmi * u[ia];
                }
            }

            for d in 0..dim {
                let dot: f32 = v_new.iter().enumerate().map(|(i, a)| a * embeddings[i][d]).sum();
                for i in 0..n_words {
                    v_new[i] -= dot * embeddings[i][d];
                }
            }

            let norm: f32 = v_new.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                v_new.iter_mut().for_each(|x| *x /= norm);
            }
            v = v_new;
        }

        for i in 0..n_words {
            embeddings[i][dim] = v[i];
        }
    }

    // ─── 4. Quantizar embeddings float → 32-bit ───
    let mut graph: HashMap<u32, u32> = HashMap::new();

    // Primeiro: embeddings morfológicos para TODAS as palavras (cobertura total)
    for (word, &(lex_id, _, _, _)) in &compiler.lexicon {
        if lex_id == 0 { continue; }
        let morph_emb = morph_vocab.word_embedding(word);
        if morph_emb != 0 {
            graph.insert(lex_id, morph_emb);
        }
    }

    // Depois: sobrepor com SVD para top-2000 (qualidade semântica)
    for (i, &(lex_id, _)) in freq_sorted.iter().enumerate() {
        let emb = &embeddings[i];

        let mut bits: u32 = 0;
        for dim in 0..n_dims {
            if emb[dim] > 0.0 {
                bits |= 1 << dim;
            }
        }

        let word = compiler.lexicon.iter()
            .find(|(_, v)| v.0 == lex_id)
            .map(|(w, _)| w.as_str())
            .unwrap_or("");

        let morph_emb = morph_vocab.word_embedding(word);
        bits ^= morph_emb;

        graph.insert(lex_id, bits);
    }

    // Copiar GRAPH para a gramática
    grammar.graph = graph.clone();
    grammar.graph_alpha = 0.3;

    // Mostrar estatísticas
    let mut decomposed_count = 0;
    let mut total_count = 0;
    for word in &word_list {
        if let Some(&(lex_id, _, _, _)) = compiler.lexicon.get(word) {
            if lex_id == 0 { continue; }
            total_count += 1;
            let (p, r, s) = morph_vocab.decompose(word);
            if p.is_some() || r.is_some() || s.is_some() {
                decomposed_count += 1;
            }
        }
    }

    println!("  GRAPH: {} embeddings (SVD + morfológico)", graph.len());
    println!("  Decomposição: {}/{} palavras ({:.1}%)",
        decomposed_count, total_count,
        decomposed_count as f32 / total_count as f32 * 100.0);

    // Mostrar exemplos de similaridade via SVD
    println!("  Top-5 similar (SVD) para 'casa':");
    if let Some(casa_id) = compiler.lexicon.get("casa").map(|v| v.0) {
        if let Some(&casa_bits) = graph.get(&casa_id) {
            let mut sims: Vec<(u32, f32)> = graph.iter()
                .filter(|&(id, _)| *id != casa_id)
                .map(|(id, bits)| {
                    let dist = (casa_bits ^ *bits).count_ones() as f32;
                    let sim = 1.0 - dist / 32.0;
                    (*id, sim)
                })
                .collect();
            sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            for &(id, sim) in sims.iter().take(5) {
                let word = compiler.lexicon.iter()
                    .find(|(_, v)| v.0 == id)
                    .map(|(w, _)| w.as_str())
                    .unwrap_or("?");
                println!("    {} ({:.2})", word, sim);
            }
        }
    }

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

    // ─── FASE 7: Gerar texto com coesão semântica ───
    println!("▸ FASE 7: Gerando texto com coesão semântica...");

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

    // Decompilar: lex_id → palavra (owned strings)
    let reverse_owned: HashMap<u32, String> = compiler.lexicon.iter()
        .map(|(w, &(id, _, _, _))| (id, w.clone()))
        .collect();

    for seed in &seeds {
        let seed_tokens = compiler.compile(seed);
        if seed_tokens.is_empty() { continue; }

        let seed_lex = seed_tokens.first().map(|t| t.lex).unwrap_or(0);
        let generated = grammar.generate_text(seed_lex, 3);

        let words: Vec<&str> = generated.iter()
            .filter(|t| t.lex != 0)
            .filter_map(|t| reverse_owned.get(&t.lex).map(|s| s.as_str()))
            .collect();

        println!("  \"{}\" → {}", seed, words.join(" "));
    }

    // Testar geração com SNN
    println!();
    println!("  Geração com SNN + coesão:");
    grammar.build_snn();
    grammar.train_snn_iterative(10, 0.001);

    for seed in &seeds[..3] {
        let seed_tokens = compiler.compile(seed);
        if seed_tokens.is_empty() { continue; }
        let generated = grammar.generate_with_snn(&seed_tokens, 20);

        let words: Vec<&str> = generated.iter()
            .filter(|t| t.lex != 0)
            .filter_map(|t| reverse_owned.get(&t.lex).map(|s| s.as_str()))
            .collect();
        println!("  \"{}\" (SNN) → {}", seed, words.join(" "));
    }

    // Testar geração retórica
    println!();
    println!("  Geração retórica (planejamento):");
    for seed in &seeds[..3] {
        let seed_tokens = compiler.compile(seed);
        if seed_tokens.is_empty() { continue; }
        let seed_lex = seed_tokens.first().map(|t| t.lex).unwrap_or(0);
        let generated = grammar.generate_text_rhetorical(seed_lex, 3);

        let words: Vec<&str> = generated.iter()
            .filter(|t| t.lex != 0)
            .filter_map(|t| reverse_owned.get(&t.lex).map(|s| s.as_str()))
            .collect();
        println!("  \"{}\" (retórico) → {}", seed, words.join(" "));
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

    grammar.train_snn_iterative(10, 0.001);

    // ─── FASE 11: Self-play ───
    println!();
    println!("▸ FASE 11: Self-play com filtro local...");

    let filter = amadeus::amadeus_m::self_play::SelfPlayFilter::new(grammar.graph.clone());
    let mut n_accepted = 0;
    let mut n_total = 0;
    let mut accepted_tokens: Vec<Token7> = Vec::new();
    let mut total_score = 0.0f32;

    // Gerar 100 textos e filtrar
    for i in 0..100 {
        let seed_lex = 1 + (i as u32 % 10); // seeds variados
        let generated = grammar.generate_text(seed_lex, 3);
        let result = filter.evaluate(&generated);

        total_score += result.score;
        n_total += 1;
        if result.passed {
            n_accepted += 1;
            accepted_tokens.extend(generated.clone());
        }

        // Log dos primeiros 10
        if i < 10 {
            let words: Vec<&str> = generated.iter()
                .filter(|t| t.lex != 0)
                .filter_map(|t| reverse_owned.get(&t.lex).map(|s| s.as_str()))
                .collect();
            println!("    [{}] {} (score={:.2} {})",
                i, words.join(" "), result.score,
                if result.passed { "✓" } else { "✗" });
        }
    }

    println!("  Gerados: {}", n_total);
    println!("  Aprovados: {} ({:.0}%)", n_accepted, n_accepted as f32 / n_total as f32 * 100.0);
    println!("  Score médio: {:.2}", total_score / n_total as f32);
    println!("  Tokens aceitos: {}", accepted_tokens.len());

    // Retreinar gramática com textos aceitos
    if !accepted_tokens.is_empty() {
        println!("  Retreinando gramática com {} tokens self-play...", accepted_tokens.len());
        grammar.train(&accepted_tokens);
    }

    // Retreinar SNN com dados expandidos
    if accepted_tokens.len() > 100 {
        println!("  Retreinando SNN...");
        grammar.build_snn();
        grammar.train_snn_iterative(5, 0.001);
    }

    println!();
    println!("═══════════════════════════════════════════════════════════");
    println!("  TREINAMENTO MISTO COMPLETO");
    println!("═══════════════════════════════════════════════════════════");
}
