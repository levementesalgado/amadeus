use amadeus::amadeus_m::triple_grammar::TripleGrammar;
use amadeus::amadeus_m::token7::Token7;
use amadeus::amadeus_m::snn;

fn main() {
    println!("╔══════════════════════════════════════════════════════╗");
    println!("║  SNN vs T3 CASCADE — FULL COMPARISON                ║");
    println!("╚══════════════════════════════════════════════════════╝\n");

    let mut grammar = TripleGrammar::new(3);
    let mut rng = fastrand::Rng::with_seed(42);

    // Gerar dados diversificados
    println!("▸ Gerando dados sintéticos...");
    let mut all_tokens: Vec<Token7> = Vec::new();
    let mut lex_counter: u32 = 1;
    let mut lex_class_map: std::collections::HashMap<u32, u8> = std::collections::HashMap::new();

    for _ in 0..150 {
        let mut seq: Vec<Token7> = Vec::new();
        for _ in 0..15 {
            let class = rng.u8(0..7);
            let gender = rng.u8(0..2);
            let number = rng.u8(0..2);
            let person = rng.u8(0..3);
            let tense = rng.u8(0..5);
            let style = rng.u16(0..3);
            let morph = amadeus::amadeus_m::token7::morph_from_class(class, gender, number, person, tense);
            let lex = lex_counter;
            lex_counter += 1;
            for _ in 0..4 {
                let t = Token7::new(lex, morph).with_style(style);
                seq.push(t);
                lex_class_map.insert(lex, class);
            }
        }
        grammar.train(&seq);
        all_tokens.extend(seq);
    }

    let mut compiler = amadeus::amadeus_m::compiler::Compiler::new("underworld");
    compiler.build_graph(&all_tokens, 3);
    grammar.graph = compiler.graph.clone();
    grammar.graph_alpha = 0.15;

    println!("  Lex_ids: {}", lex_counter - 1);
    println!("  T3 exact={}", grammar.lexicon.len());

    // Construir SNN
    println!("\n▸ Construindo SNN...");
    grammar.build_snn();
    let snn = grammar.snn.as_ref().unwrap();
    println!("  Input={}, Hidden={}, Output={}", snn.input_neurons.len(), snn.hidden_neurons.len(), snn.output_neurons.len());

    // Comparar rankings
    println!("\n▸ Comparando rankings...\n");

    let t3_keys: Vec<(u16, u8, u16)> = grammar.lexicon.keys().copied().collect();
    let lex_class_map = grammar.lex_to_class.clone();

    let mut snn_top1 = 0usize;
    let mut snn_top3 = 0usize;
    let mut snn_top5 = 0usize;
    let mut cascade_top1 = 0usize;
    let mut total = 0usize;
    let mut both_same = 0usize;
    let mut example_count = 0usize;

    for &key in t3_keys.iter().take(30) {
        let morph = key.0;
        let syn_func = key.1;
        let style = key.2;
        let cls = *lex_class_map.values().next().unwrap_or(&0);

        if let Some(candidates) = grammar.lexicon.get(&key) {
            if let Some((&next_id, _)) = candidates.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()) {
                let candidate = next_id;

                let cascade_result = grammar.sample_lex_full(morph, syn_func, style, cls, candidate);

                let mut snn_mut = grammar.snn.take().unwrap();
                let enc = grammar.snn_enc.as_ref().unwrap();
                let rankings = snn::infer_snn(&mut snn_mut, enc, morph, syn_func, style, &mut grammar.rng);
                grammar.snn = Some(snn_mut);

                let top1_ids: Vec<u32> = rankings.iter().take(1).map(|&(id, _)| id).collect();
                let top3_ids: Vec<u32> = rankings.iter().take(3).map(|&(id, _)| id).collect();
                let top5_ids: Vec<u32> = rankings.iter().take(5).map(|&(id, _)| id).collect();

                total += 1;
                if cascade_result == next_id { cascade_top1 += 1; }
                if snn_result_matches(&rankings, next_id, 1) { snn_top1 += 1; }
                if snn_result_matches(&rankings, next_id, 3) { snn_top3 += 1; }
                if snn_result_matches(&rankings, next_id, 5) { snn_top5 += 1; }
                if cascade_result == next_id && snn_result_matches(&rankings, next_id, 1) { both_same += 1; }

                if example_count < 3 {
                    example_count += 1;
                    println!("  Exemplo {}: morph={:#06x} syn={} style={}", example_count, morph, syn_func, style);
                    println!("    Golden:    lex_id={}", next_id);
                    println!("    Cascade:   lex_id={}", cascade_result);
                    println!("    SNN top-5:");
                    for (i, (id, score)) in rankings.iter().take(5).enumerate() {
                        let marker = if *id == next_id { " ← GOLDEN" } else { "" };
                        println!("      {:>2}. lex_id={:>5}  score={:.3}{}", i + 1, id, score, marker);
                    }
                    println!();
                }
            }
        }
    }

    println!("  ┌──────────────────────────────────────────────┐");
    println!("  │ Total:            {:>6}                   │", total);
    println!("  │ Cascade top-1:    {:>6} ({:>5.1}%)           │", cascade_top1, cascade_top1 as f64 / total as f64 * 100.0);
    println!("  │ SNN top-1:        {:>6} ({:>5.1}%)           │", snn_top1, snn_top1 as f64 / total as f64 * 100.0);
    println!("  │ SNN top-3:        {:>6} ({:>5.1}%)           │", snn_top3, snn_top3 as f64 / total as f64 * 100.0);
    println!("  │ SNN top-5:        {:>6} ({:>5.1}%)           │", snn_top5, snn_top5 as f64 / total as f64 * 100.0);
    println!("  │ Both top-1:       {:>6} ({:>5.1}%)           │", both_same, both_same as f64 / total as f64 * 100.0);
    println!("  └──────────────────────────────────────────────┘");

    // GGUF check
    println!("\n▸ Verificando GGUF roundtrip...");
    grammar.save_gguf("/tmp/amadeus_snn_test.grammar.gguf", &compiler.lexicon, &compiler.roots).expect("falha GGUF save");
    let mut loaded = TripleGrammar::new(3);
    loaded.load_gguf("/tmp/amadeus_snn_test.grammar.gguf").expect("falha GGUF load");
    let _ = std::fs::remove_file("/tmp/amadeus_snn_test.grammar.gguf");
    println!("  GGUF roundtrip OK");

    println!("\n✅ Comparação completa!");
}

fn snn_result_matches(rankings: &[(u32, f32)], target: u32, top_k: usize) -> bool {
    rankings.iter().take(top_k).any(|(id, _)| *id == target)
}
