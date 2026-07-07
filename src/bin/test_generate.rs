use amadeus::amadeus_m::triple_grammar::TripleGrammar;
use amadeus::amadeus_m::compiler::Compiler;
use amadeus::amadeus_m::token7::Token7;
use amadeus::amadeus_m::snn;
use std::collections::HashMap;

fn main() {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  AMADEUS — GERAÇÃO E ANÁLISE DE TEXTO                  ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    // ═══════════════════════════════════════════════════════════
    // 1. Treinar gramática com dados sintéticos estruturados
    // ═══════════════════════════════════════════════════════════
    println!("▸ FASE 1: Treinando gramática...");
    let mut grammar = TripleGrammar::new(3);
    let mut rng = fastrand::Rng::with_seed(42);

    // Simular frases portuguesas com estrutura real
    let frases: Vec<Vec<(&str, u8, u8, u8, u8, u8)>> = vec![
        // "O gato preto dormiu no sofá"
        vec![("o", 3,0,0,0,0), ("gato", 0,0,0,2,0), ("preto", 2,0,0,2,0), ("dormiu", 1,0,0,3,2), ("no", 4,0,0,0,0), ("sofá", 0,0,0,2,0)],
        // "A mulher bonita leu o livro"
        vec![("a", 3,1,0,0,0), ("mulher", 0,1,0,2,0), ("bonita", 2,1,0,2,0), ("leu", 1,0,0,3,2), ("o", 3,0,0,0,0), ("livro", 0,0,0,2,0)],
        // "Os meninos correram no parque"
        vec![("os", 3,0,1,0,0), ("meninos", 0,0,1,2,0), ("correram", 1,0,1,3,2), ("no", 4,0,0,0,0), ("parque", 0,0,0,2,0)],
        // "Eu gosto de música"
        vec![("eu", 6,0,0,0,0), ("gosto", 1,0,0,1,0), ("de", 4,0,0,0,0), ("música", 0,1,0,2,0)],
        // "O sol brilha forte"
        vec![("o", 3,0,0,0,0), ("sol", 0,0,0,2,0), ("brilha", 1,0,0,1,0), ("forte", 2,0,0,2,0)],
        // "A chuva caiu ontem"
        vec![("a", 3,1,0,0,0), ("chuva", 0,1,0,2,0), ("caiu", 1,0,0,3,2), ("ontem", 6,0,0,0,0)],
        // "Nós estudamos português"
        vec![("nós", 6,0,1,0,0), ("estudamos", 1,0,1,1,0), ("português", 0,0,0,2,0)],
        // "O menino comeu a maçã"
        vec![("o", 3,0,0,0,0), ("menino", 0,0,0,2,0), ("comeu", 1,0,0,3,2), ("a", 3,1,0,0,0), ("maçã", 0,1,0,2,0)],
        // "Ela viajou para o sul"
        vec![("ela", 6,1,0,0,0), ("viajou", 1,0,0,3,2), ("para", 4,0,0,0,0), ("o", 3,0,0,0,0), ("sul", 0,0,0,2,0)],
        // "O animal dormiu bem"
        vec![("o", 3,0,0,0,0), ("animal", 0,0,0,2,0), ("dormiu", 1,0,0,3,2), ("bem", 6,0,0,0,0)],
    ];

    // Compilar e treinar
    let mut lex_counter: u32 = 1;
    let mut lex_map: HashMap<String, u32> = HashMap::new();
    let mut all_tokens: Vec<Token7> = Vec::new();

    for _ in 0..50 { // repetir 50x para Popular tabelas
        for frase in &frases {
            let mut seq: Vec<Token7> = Vec::new();
            for (word, class, gender, number, tense, person) in frase {
                let lex = *lex_map.entry(word.to_string()).or_insert_with(|| {
                    let id = lex_counter;
                    lex_counter += 1;
                    id
                });
                let morph = amadeus::amadeus_m::token7::morph_from_class(*class, *gender, *number, *person, *tense);
                let style = 0u16; // neutro
                seq.push(Token7::new(lex, morph).with_style(style));
            }
            grammar.train(&seq);
            all_tokens.extend(seq);
        }
    }

    let mut compiler = Compiler::new("underworld");
    compiler.build_graph(&all_tokens, 3);
    grammar.graph = compiler.graph.clone();
    grammar.graph_alpha = 0.2;

    // Popula o léxico do compiler para decompile funcionar
    for (word, &id) in &lex_map {
        let class = 0u8; // simplificado
        let morph = amadeus::amadeus_m::token7::morph_from_class(class, 0, 0, 0, 0);
        compiler.lexicon.insert(word.clone(), (id, morph, class, 0));
        if compiler.roots.len() <= id as usize {
            compiler.roots.resize(id as usize + 1, String::new());
        }
        compiler.roots[id as usize] = word.clone();
    }

    println!("  Léxico: {} palavras", lex_map.len());
    println!("  T2: {} contextos", grammar.agreement.len());
    println!("  T3: {} contextos", grammar.lexicon.len());
    println!("  GRAPH: {} embeddings", grammar.graph.len());

    // Construir SNN
    grammar.build_snn();
    let snn = grammar.snn.as_ref().unwrap();
    println!("  SNN: {}→{}→{} neurons", snn.input_neurons.len(), snn.hidden_neurons.len(), snn.output_neurons.len());

    // ═══════════════════════════════════════════════════════════
    // 2. Gerar textos
    // ═══════════════════════════════════════════════════════════
    println!("\n▸ FASE 2: Gerando textos...\n");

    let sementes = vec![
        ("o gato", "Cena doméstica"),
        ("a mulher", "Ação humana"),
        ("os meninos", "Ação coletiva"),
        ("eu gosto", "Expressão pessoal"),
        ("o sol", "Natureza"),
    ];

    grammar.temperature = 0.8;
    grammar.exploration_rate = 0.05;

    for (seed_text, descricao) in &sementes {
        println!("  ── {} ({}) ──", seed_text, descricao);

        // Compilar semente
        let seed_tokens = compiler.compile(seed_text);
        if seed_tokens.is_empty() {
            println!("    (semente não compilou)\n");
            continue;
        }

        // Gerar com cascade legada
        let generated_cascade = grammar.generate(&seed_tokens, 12);
        let output_cascade = decompile_simple(&generated_cascade, &lex_map);
        println!("    Cascade: {}", output_cascade);

        // Gerar com SNN
        let generated_snn = grammar.generate_with_snn(&seed_tokens, 12);
        let output_snn = decompile_simple(&generated_snn, &lex_map);
        println!("    SNN:     {}", output_snn);

        // Analisar tokens gerados
        println!("    Tokens cascade:");
        for (i, tok) in generated_cascade.iter().enumerate() {
            if tok.lex != 0 {
                let word = lex_map.iter().find(|(_, id)| **id == tok.lex).map(|(w, _)| w.as_str()).unwrap_or("?");
                println!("      [{:>2}] lex={:>3} morph={:#06x} class={} word={}", i, tok.lex, tok.morph, tok.morph_class(), word);
            }
        }
        println!();
    }

    // ═══════════════════════════════════════════════════════════
    // 3. Análise SNN vs Cascade
    // ═══════════════════════════════════════════════════════════
    println!("▸ FASE 3: Análise comparativa SNN vs Cascade...\n");

    let mut cascade_classes: HashMap<u8, usize> = HashMap::new();
    let mut snn_classes: HashMap<u8, usize> = HashMap::new();
    let mut total_tokens = 0usize;

    for _ in 0..20 {
        let seed_tokens = compiler.compile("o gato");
        if seed_tokens.is_empty() { continue; }

        let gen_c = grammar.generate(&seed_tokens, 10);
        let gen_s = grammar.generate_with_snn(&seed_tokens, 10);

        for tok in &gen_c {
            *cascade_classes.entry(tok.morph_class()).or_insert(0) += 1;
            total_tokens += 1;
        }
        for tok in &gen_s {
            *snn_classes.entry(tok.morph_class()).or_insert(0) += 1;
        }
    }

    println!("  Distribuição de classes geradas (20 seeds × 10 tokens):");
    println!("  {:>8} {:>10} {:>10}", "Classe", "Cascade", "SNN");
    let classes = vec!["SUBST", "VERBO", "ADJ", "ART", "PREP", "PONT", "OUT"];
    for (i, name) in classes.iter().enumerate() {
        let c_count = cascade_classes.get(&(i as u8)).unwrap_or(&0);
        let s_count = snn_classes.get(&(i as u8)).unwrap_or(&0);
        println!("  {:>8} {:>10} {:>10}", name, c_count, s_count);
    }

    // ═══════════════════════════════════════════════════════════
    // 4. Análise de concordância
    // ═══════════════════════════════════════════════════════════
    println!("\n▸ FASE 4: Análise de concordância...\n");

    let mut concordancia_ok = 0usize;
    let mut concordancia_total = 0usize;

    for _ in 0..20 {
        let seed_tokens = compiler.compile("a mulher");
        if seed_tokens.is_empty() { continue; }

        let generated = grammar.generate(&seed_tokens, 8);

        // Verificar concordância: artigo feminino → substantivo feminino
        for window in generated.windows(3) {
            if window[0].morph_class() == 3 && ((window[0].morph >> 3) & 1) == 1 { // artigo feminino
                concordancia_total += 1;
                if window[1].morph_class() == 0 && ((window[1].morph >> 3) & 1) == 1 { // subst feminino
                    concordancia_ok += 1;
                }
            }
        }
    }

    if concordancia_total > 0 {
        println!("  Concordância artigo-substantivo (feminino): {}/{} ({:.1}%)",
            concordancia_ok, concordancia_total,
            concordancia_ok as f64 / concordancia_total as f64 * 100.0);
    } else {
        println!("  (sem exemplos de concordância para analisar)");
    }

    // ═══════════════════════════════════════════════════════════
    // 5. Benchmark
    // ═══════════════════════════════════════════════════════════
    println!("\n▸ FASE 5: Benchmark...\n");

    let seed_tokens = compiler.compile("o mundo");
    if !seed_tokens.is_empty() {
        let n_iter = 100;

        let start = std::time::Instant::now();
        for _ in 0..n_iter {
            let _ = grammar.generate(&seed_tokens, 10);
        }
        let cascade_time = start.elapsed();

        let start = std::time::Instant::now();
        for _ in 0..n_iter {
            let _ = grammar.generate_with_snn(&seed_tokens, 10);
        }
        let snn_time = start.elapsed();

        let tokens_per_iter = 10 + seed_tokens.len();
        println!("  Cascade: {:?} ({:.1} tok/s)", cascade_time,
            (n_iter * tokens_per_iter) as f64 / cascade_time.as_secs_f64());
        println!("  SNN:     {:?} ({:.1} tok/s)", snn_time,
            (n_iter * tokens_per_iter) as f64 / snn_time.as_secs_f64());
        println!("  Speedup: {:.2}x", cascade_time.as_nanos() as f64 / snn_time.as_nanos() as f64);
    }

    println!("\n═══════════════════════════════════════════════════════════");
    println!("  GERAÇÃO E ANÁLISE COMPLETAS");
    println!("═══════════════════════════════════════════════════════════\n");
}

/// Decompilar tokens para texto (simplificado)
fn decompile_simple(tokens: &[Token7], lex_map: &HashMap<String, u32>) -> String {
    let reverse: HashMap<u32, &str> = lex_map.iter().map(|(w, &id)| (id, w.as_str())).collect();
    let words: Vec<&str> = tokens.iter()
        .filter(|t| t.lex != 0)
        .filter_map(|t| reverse.get(&t.lex).copied())
        .collect();
    words.join(" ")
}
