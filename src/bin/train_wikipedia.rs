use amadeus::amadeus_m::triple_grammar::TripleGrammar;
use amadeus::amadeus_m::token7::Token7;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Compilador simplificado que aprende vocabulário diretamente do texto
struct SimpleCompiler {
    lexicon: HashMap<String, (u32, u16, u8, u16)>, // word -> (id, morph, class, style)
    roots: Vec<String>,
    next_id: u32,
}

impl SimpleCompiler {
    fn new() -> Self {
        let mut c = Self {
            lexicon: HashMap::new(),
            roots: vec![String::new()],
            next_id: 1,
        };

        // Adicionar pontuação (class 6 = PONTUACAO)
        for p in &[",", ".", "!", "?", ";", ":", "—", "-", "\"", "(", ")", "..."] {
            let morph = 6u16; // PONTUACAO
            c.lexicon.insert(p.to_string(), (0, morph, 6, 0));
        }

        c
    }

    fn compile(&mut self, text: &str) -> Vec<Token7> {
        let mut tokens = Vec::new();
        let words: Vec<&str> = text.split(|c: char| c.is_whitespace() || c == ',' || c == '.' || c == '!' || c == '?' || c == ';' || c == ':').collect();

        for w in words {
            let w = w.trim();
            if w.is_empty() { continue; }

            // Verificar pontuação
            if let Some(&(id, morph, class, style)) = self.lexicon.get(w) {
                tokens.push(Token7::new(id, morph).with_style(style));
                continue;
            }

            // Palavra desconhecida - aprender
            let lower = w.to_lowercase();
            if let Some(&(id, morph, class, style)) = self.lexicon.get(&lower) {
                tokens.push(Token7::new(id, morph).with_style(style));
                continue;
            }

            // Inferir classe morfológica
            let (id, morph, class, style) = self.infer_word(&lower);
            tokens.push(Token7::new(id, morph).with_style(style));
        }

        tokens
    }

    fn infer_word(&mut self, word: &str) -> (u32, u16, u8, u16) {
        let id = self.next_id;
        self.next_id += 1;

        // Salvar raiz
        if id as usize >= self.roots.len() {
            self.roots.resize(id as usize + 1, String::new());
        }
        self.roots[id as usize] = word.to_string();

        // Inferir classe morfológica
        let (class, morph_bits) = if word.ends_with("ção") || word.ends_with("ões") || word.ends_with("mento") || word.ends_with("ismo") || word.ends_with("idade") || word.ends_with("agem") {
            (0, 0b0000_0001u16) // SUBST (sufixos nominais)
        } else if word.ends_with("mente") {
            (4, 0b0000_0010u16) // ADVERB
        } else if word.ends_with("oso") || word.ends_with("osa") || word.ends_with("vel") || word.ends_with("al") || word.ends_with("ível") || word.ends_with("ário") || word.ends_with("ário") || word.ends_with("ico") || word.ends_with("ica") || word.ends_with("nte") || word.ends_with("nte") {
            (2, 0b0000_0100u16) // ADJ (muitos sufixos)
        } else if word.ends_with("ar") || word.ends_with("ear") || word.ends_with("izar") || word.ends_with("ar") || word.ends_with("or") {
            (1, 0b0000_1000u16) // VERB (infinitivo/gerúndio)
        } else if word.ends_with("ou") || word.ends_with("iu") || word.ends_with("eu") || word.ends_with("amos") || word.ends_with("eis") || word.ends_with("em") {
            (1, 0b0001_0000u16) // VERB (passado/presente)
        } else if word.ends_with("ado") || word.ends_with("ido") || word.ends_with("ito") || word.ends_with("sto") {
            (1, 0b0010_0000u16) // VERB (particípio)
        } else if word == "o" || word == "os" {
            (3, 0b0100_0000u16) // ART (masculino)
        } else if word == "a" || word == "as" {
            (3, 0b1000_0000u16) // ART (feminino)
        } else if word == "um" || word == "uns" {
            (3, 0b0100_0001u16) // ART (indefinido masc)
        } else if word == "uma" || word == "umas" {
            (3, 0b1000_0001u16) // ART (indefinido fem)
        } else if word == "de" || word == "do" || word == "da" || word == "dos" || word == "das" {
            (5, 0b0000_0011u16) // PREP (contração)
        } else if word == "em" || word == "no" || word == "na" || word == "nos" || word == "nas" {
            (5, 0b0000_0110u16) // PREP (locução)
        } else if word == "por" || word == "para" || word == "com" || word == "sem" || word == "sob" {
            (5, 0b0000_1001u16) // PREP (simples)
        } else if word == "e" || word == "ou" || word == "mas" || word == "porém" || word == "então" {
            (5, 0b0000_1100u16) // CONJ
        } else if word == "que" || word == "se" || word == "como" || word == "quando" {
            (5, 0b0000_1111u16) // CONJ (subord)
        } else if word == "eu" || word == "tu" || word == "ele" || word == "ela" || word == "nós" || word == "vós" || word == "eles" || word == "elas" {
            (3, 0b0001_0000u16) // PRON
        } else if word == "isto" || word == "isso" || word == "aquilo" || word == "este" || word == "esse" || word == "aquele" {
            (3, 0b0010_0000u16) // PRON (demostrativo)
        } else if word == "bem" || word == "mal" || word == "muito" || word == "pouco" || word == "mais" || word == "menos" {
            (4, 0b0001_0001u16) // ADVERB (grau)
        } else if word == "aqui" || word == "aí" || word == "ali" || word == "lá" || word == "cá" || word == "onde" {
            (4, 0b0010_0010u16) // ADVERB (lugar)
        } else if word == "agora" || word == "então" || word == "depois" || word == "antes" || word == "sempre" || word == "nunca" {
            (4, 0b0011_0011u16) // ADVERB (tempo)
        } else {
            // Default: SUBST com bits baseados no ID
            let bits = (id as u16).wrapping_mul(0x9E37) & 0x00FF;
            (0, bits)
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
    println!("║  AMADEUS — TREINAMENTO COM WIKIPÉDIA PT               ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();

    // ─── FASE 1: Coletar textos ───
    println!("▸ FASE 1: Coletando textos da Wikipédia PT...");

    let wiki_dir = "training/wikipedia";
    let mut all_text = String::new();
    let mut file_count = 0;

    if Path::new(wiki_dir).exists() {
        for entry in fs::read_dir(wiki_dir).unwrap().flatten() {
            let path = entry.path();
            if path.extension().map(|e| e == "txt").unwrap_or(false) {
                if let Ok(content) = fs::read_to_string(&path) {
                    let file_name = path.file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("?");
                    println!("  → {} ({} KB)", file_name, content.len() / 1024);
                    all_text.push_str(&content);
                    all_text.push('\n');
                    file_count += 1;
                }
            }
        }
    }

    println!("  {} arquivos, {} caracteres totais", file_count, all_text.len());
    println!();

    // ─── FASE 2: Compilar tokens ───
    println!("▸ FASE 2: Compilando tokens...");

    let mut compiler = SimpleCompiler::new();

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

    // ─── FASE 3: Treinar gramática ───
    println!("▸ FASE 3: Treinando gramática...");

    let mut grammar = TripleGrammar::new(3);
    let tokens_with_deps = amadeus::amadeus_m::syntax::assign_dependencies(&all_tokens);

    // Treinar em lotes
    let batch_size = 100;
    for (i, batch) in tokens_with_deps.chunks(batch_size).enumerate() {
        grammar.train(batch);
        if (i + 1) % 100 == 0 {
            println!("  ... {} tokens processados", (i + 1) * batch_size);
        }
    }

    println!("  Treinamento concluído");
    println!();

    // ─── FASE 4: Construir GRAPH embeddings ───
    println!("▸ FASE 4: Construindo GRAPH embeddings...");

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

    for &lex_id in &lex_ids {
        let mut embedding: u32 = 0;

        // Somar projeções dos contextos
        for ((a, b), &count) in &cooccurrence {
            if *a == lex_id || *b == lex_id {
                let other = if *a == lex_id { *b } else { *a };
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

    // ─── FASE 5: Métricas ───
    println!("▸ FASE 5: Métricas do treinamento...");

    let lex_count = grammar.lexicon.len();
    let t2_count = grammar.agreement.len();
    let t3_count = grammar.lexicon.len();
    let graph_count = grammar.graph.len();

    println!("  Léxico: {} entradas", lex_count);
    println!("  T2 (concordância): {} padrões", t2_count);
    println!("  T3 (seleção): {} entradas", t3_count);
    println!("  GRAPH: {} embeddings", graph_count);
    println!();

    // ─── FASE 6: Gerar texto com GRAPH modulation ───
    println!("▸ FASE 6: Gerando texto com GRAPH modulation...");

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

    // ─── FASE 7: Salvar ───
    println!("▸ FASE 7: Salvando gramática treinada...");

    let bin_path = "training/wikipedia_grammar.bin";
    let gguf_path = "training/wikipedia_grammar.gguf";

    grammar.save(bin_path).unwrap();
    println!("  BIN: {} bytes", fs::metadata(bin_path).unwrap().len());

    grammar.save_gguf(gguf_path, &compiler.lexicon, &compiler.roots).unwrap();
    println!("  GGUF: {} bytes", fs::metadata(gguf_path).unwrap().len());

    println!();

    // ─── FASE 8: Treinar SNN com dados reais ───
    println!("▸ FASE 8: Treinando SNN com dados reais...");

    // Construir SNN a partir da gramática treinada
    grammar.build_snn();

    println!("  SNN construída a partir da gramática");
    if let Some(ref snn) = grammar.snn {
        println!("  Pesos SNN: {} synapses IH, {} synapses HO",
            snn.synapses_ih.len(), snn.synapses_ho.len());
    }

    println!();
    println!("═══════════════════════════════════════════════════════════");
    println!("  TREINAMENTO COM WIKIPÉDIA COMPLETO");
    println!("═══════════════════════════════════════════════════════════");
}
