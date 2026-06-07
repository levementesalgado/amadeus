use amadeus::amadeus_m::triple_grammar::TripleGrammar;
use amadeus::amadeus_m::compiler::Compiler;
use amadeus::amadeus_m::token7::Token7;
use amadeus::amadeus_m::pcfg::generate_many;

fn main() {
    println!("╔══════════════════════════════════════╗");
    println!("║  GGUF ROUNDTRIP TEST — AMADEUS       ║");
    println!("╚══════════════════════════════════════╝\n");

    // ─── 1. Train grammar with synthetic data ───
    println!("▸ Treinando gramática com PCFG sintético...");
    let compiler = Compiler::new("underworld");
    let mut grammar = TripleGrammar::new(3);

    let seqs = generate_many(&compiler, 500, 42);
    let mut total_tokens = 0usize;
    for seq in &seqs {
        if seq.len() >= 2 {
            grammar.train(seq);
            total_tokens += seq.len();
        }
    }

    // Build GRAPH
    let all_tokens: Vec<Token7> = seqs.iter().flat_map(|s| s.iter().copied()).collect();
    let mut compiler_for_graph = compiler;
    compiler_for_graph.build_graph(&all_tokens, 3);
    grammar.graph = compiler_for_graph.graph.clone();
    grammar.graph_alpha = 0.15;

    println!("  Tokens treinados: {}", total_tokens);
    println!("  C_O={}, C_F={}, C_P={}, C_T={}",
        grammar.hier.clause.table.len(),
        grammar.hier.sentence.table.len(),
        grammar.hier.paragraph.table.len(),
        grammar.hier.text.table.len());
    println!("  T2={}, T2+={}, T3={}, CSS={}, CS={}, CST={}, Class={}",
        grammar.agreement.len(),
        grammar.agreement_gparent.len(),
        grammar.lexicon.len(),
        grammar.lexicon_cls_syn_style.len(),
        grammar.lexicon_cls_syn.len(),
        grammar.lexicon_cls_style.len(),
        grammar.lexicon_class.len());
    println!("  GRAPH={}, lex_to_class={}", grammar.graph.len(), grammar.lex_to_class.len());
    println!("  Compilador: {} entradas no léxico", compiler_for_graph.lexicon.len());

    // ─── 2. Save to GGUF ───
    let gguf_path = "/tmp/amadeus_test.grammar.gguf";
    println!("\n▸ Salvando em GGUF: {}", gguf_path);
    grammar.save_gguf(gguf_path, &compiler_for_graph.lexicon, &compiler_for_graph.roots)
        .expect("falha ao salvar GGUF");
    let gguf_size = std::fs::metadata(gguf_path).unwrap().len();
    println!("  Tamanho: {:.1} KB", gguf_size as f64 / 1024.0);

    // Also save binary for comparison
    let bin_path = "/tmp/amadeus_test.grammar.bin";
    grammar.save(bin_path).expect("falha ao salvar binário");
    let bin_size = std::fs::metadata(bin_path).unwrap().len();
    println!("  Binário equivalente: {:.1} KB", bin_size as f64 / 1024.0);

    // ─── 3. Load from GGUF into fresh grammar ───
    println!("\n▸ Carregando GGUF em gramática nova...");
    let mut loaded = TripleGrammar::new(3);
    loaded.load_gguf(gguf_path).expect("falha ao carregar GGUF");

    // ─── 4. Verify data integrity ───
    println!("\n▸ Verificando integridade dos dados...\n");
    let mut errors = 0usize;

    // Config
    check("syntax_order", grammar.syntax_order == loaded.syntax_order, &mut errors);
    check("temperature", (grammar.temperature - loaded.temperature).abs() < 1e-6, &mut errors);
    check("exploration_rate", (grammar.exploration_rate - loaded.exploration_rate).abs() < 1e-6, &mut errors);
    check("graph_alpha", (grammar.graph_alpha - loaded.graph_alpha).abs() < 1e-6, &mut errors);

    // CUBO sizes
    check("C_O size", grammar.hier.clause.table.len() == loaded.hier.clause.table.len(), &mut errors);
    check("C_F size", grammar.hier.sentence.table.len() == loaded.hier.sentence.table.len(), &mut errors);
    check("C_P size", grammar.hier.paragraph.table.len() == loaded.hier.paragraph.table.len(), &mut errors);
    check("C_T size", grammar.hier.text.table.len() == loaded.hier.text.table.len(), &mut errors);

    // T2 sizes
    check("T2 head size", grammar.agreement.len() == loaded.agreement.len(), &mut errors);
    check("T2 gparent size", grammar.agreement_gparent.len() == loaded.agreement_gparent.len(), &mut errors);
    check("T2 class size", grammar.agreement_class.len() == loaded.agreement_class.len(), &mut errors);

    // T3 sizes
    check("T3 exact size", grammar.lexicon.len() == loaded.lexicon.len(), &mut errors);
    check("T3 CSS size", grammar.lexicon_cls_syn_style.len() == loaded.lexicon_cls_syn_style.len(), &mut errors);
    check("T3 CS size", grammar.lexicon_cls_syn.len() == loaded.lexicon_cls_syn.len(), &mut errors);
    check("T3 CST size", grammar.lexicon_cls_style.len() == loaded.lexicon_cls_style.len(), &mut errors);
    check("T3 Class size", grammar.lexicon_class.len() == loaded.lexicon_class.len(), &mut errors);

    // GRAPH
    check("GRAPH size", grammar.graph.len() == loaded.graph.len(), &mut errors);

    // Spot-check some T2 entries
    let t2_match = grammar.agreement.iter().all(|(k, v)| {
        if let Some(loaded_v) = loaded.agreement.get(k) {
            v.len() == loaded_v.len() && v.iter().all(|(kk, vv)| {
                loaded_v.get(kk).map(|lv| (vv - lv).abs() < 0.01).unwrap_or(false)
            })
        } else {
            false
        }
    });
    check("T2 head values", t2_match, &mut errors);

    // Spot-check some T3 entries
    let t3_match = grammar.lexicon.iter().all(|(k, v)| {
        if let Some(loaded_v) = loaded.lexicon.get(k) {
            v.len() == loaded_v.len() && v.iter().all(|(kk, vv)| {
                loaded_v.get(kk).map(|lv| (vv - lv).abs() < 0.01).unwrap_or(false)
            })
        } else {
            false
        }
    });
    check("T3 exact values", t3_match, &mut errors);

    // Spot-check GRAPH
    let graph_match = grammar.graph.iter().all(|(k, v)| {
        loaded.graph.get(k).map(|lv| v == lv).unwrap_or(false)
    });
    check("GRAPH values", graph_match, &mut errors);

    // CUBO lambda check
    let clause_lambdas_match = grammar.hier.clause.lambda.len() == loaded.hier.clause.lambda.len()
        && grammar.hier.clause.lambda.iter().zip(loaded.hier.clause.lambda.iter())
            .all(|(a, b)| (a - b).abs() < 1e-5);
    check("C_O lambdas", clause_lambdas_match, &mut errors);

    // CUBO unigram check
    let unigram_match = grammar.hier.clause.unigram.len() == loaded.hier.clause.unigram.len()
        && grammar.hier.clause.unigram.iter().all(|(k, v)| {
            loaded.hier.clause.unigram.get(k).map(|lv| (v - lv).abs() < 1e-5).unwrap_or(false)
        });
    check("C_O unigram", unigram_match, &mut errors);

    // ─── 5. Summary ───
    println!("{}", "─".repeat(50));
    if errors == 0 {
        println!("✅ TODOS OS TESTES PASSARAM — roundtrip GGUF íntegro");
    } else {
        println!("❌ {} ERROS encontrados", errors);
    }
    println!("{}", "─".repeat(50));

    // Cleanup
    let _ = std::fs::remove_file(gguf_path);
    let _ = std::fs::remove_file(bin_path);
}

fn check(name: &str, ok: bool, errors: &mut usize) {
    if ok {
        println!("  ✓ {}", name);
    } else {
        println!("  ✗ {} — FALHA", name);
        *errors += 1;
    }
}
