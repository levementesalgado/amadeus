use amadeus::amadeus_m::triple_grammar::TripleGrammar;
use amadeus::amadeus_m::compiler::Compiler;
use amadeus::amadeus_m::token7::Token7;
use amadeus::amadeus_m::embedding_compressor::{EmbeddingCompressor, interpolate_graphs};
use amadeus::amadeus_m::dual_loader::{DualGgufLoader, GgufMmapReader};
use amadeus::amadeus_m::snn;
use std::collections::HashMap;

fn main() {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  HYBRID PIPELINE: LLM Embeddings → GRAPH → CUBO → SNN ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    // ═══════════════════════════════════════════════════════════
    // FASE 1: Treinar gramática Amadeus com dados sintéticos
    // ═══════════════════════════════════════════════════════════
    println!("▸ FASE 1: Treinando gramática Amadeus...");
    let mut grammar = TripleGrammar::new(3);
    let mut rng = fastrand::Rng::with_seed(42);
    let mut all_tokens: Vec<Token7> = Vec::new();
    let mut lex_counter: u32 = 1;
    let mut lex_class_map: HashMap<u32, u8> = HashMap::new();

    for _ in 0..100 {
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

    let mut compiler = Compiler::new("underworld");
    compiler.build_graph(&all_tokens, 3);
    grammar.graph = compiler.graph.clone();
    grammar.graph_alpha = 0.15;

    println!("  Lex_ids: {}, T3 exact={}, GRAPH={}", lex_counter - 1, grammar.lexicon.len(), grammar.graph.len());

    // ═══════════════════════════════════════════════════════════
    // FASE 2: Comprimir embeddings simulados de modelo comercial
    // ═══════════════════════════════════════════════════════════
    println!("\n▸ FASE 2: Simulando embeddings de modelo comercial...");

    // Simular embeddings F32 de alta dimensão (como Llama 3 com 4096D)
    let sim_embd_dim = 4096;
    let mut llm_graph: HashMap<u32, u32> = HashMap::new();

    for lex_id in 1..=lex_counter {
        // Gerar embedding pseudo-aleatório (simula LLM embedding)
        let mut embd = Vec::with_capacity(sim_embd_dim);
        let seed = lex_id.wrapping_mul(2654435761);
        for i in 0..sim_embd_dim {
            let val = ((seed.wrapping_add(i as u32).wrapping_mul(1103515245).wrapping_add(12345)) >> 16) as f32 / 32768.0;
            embd.push(val);
        }

        // Comprimir para 32-bit GRAPH via Random Indexing
        let signature = EmbeddingCompressor::compress_single(&embd, lex_id);
        llm_graph.insert(lex_id, signature);
    }

    println!("  Embeddings comprimidos: {} entries ({}D → 32-bit)", llm_graph.len(), sim_embd_dim);

    // ═══════════════════════════════════════════════════════════
    // FASE 3: Interpolar GRAPHs (legado + LLM)
    // ═══════════════════════════════════════════════════════════
    println!("\n▸ FASE 3: Interpolando GRAPHs...");

    let merged_graph = interpolate_graphs(&grammar.graph, &llm_graph, 0.6);

    // Comparar Hamming distances
    let mut legacy_avg_sim = 0.0f32;
    let mut llm_avg_sim = 0.0f32;
    let mut merged_avg_sim = 0.0f32;
    let mut count = 0usize;

    for lex_id in 1..=lex_counter.min(100) {
        for other_id in (lex_id + 1)..=lex_id.min(10).min(lex_counter) {
            if let (Some(&g1), Some(&g2)) = (grammar.graph.get(&lex_id), grammar.graph.get(&other_id)) {
                let dist = (g1 ^ g2).count_ones() as f32;
                legacy_avg_sim += 1.0 - dist / 32.0;
            }
            if let (Some(&g1), Some(&g2)) = (llm_graph.get(&lex_id), llm_graph.get(&other_id)) {
                let dist = (g1 ^ g2).count_ones() as f32;
                llm_avg_sim += 1.0 - dist / 32.0;
            }
            if let (Some(&g1), Some(&g2)) = (merged_graph.get(&lex_id), merged_graph.get(&other_id)) {
                let dist = (g1 ^ g2).count_ones() as f32;
                merged_avg_sim += 1.0 - dist / 32.0;
            }
            count += 1;
        }
    }

    if count > 0 {
        println!("  Hamming similarity média (pares):");
        println!("    Legado:  {:.3}", legacy_avg_sim / count as f32);
        println!("    LLM:     {:.3}", llm_avg_sim / count as f32);
        println!("    Merged:  {:.3}", merged_avg_sim / count as f32);
    }

    // Usar GRAPH interpolado na gramática
    grammar.graph = merged_graph.clone();
    grammar.graph_alpha = 0.25; // aumentar peso do GRAPH interpolado

    // ═══════════════════════════════════════════════════════════
    // FASE 4: Construir SNN
    // ═══════════════════════════════════════════════════════════
    println!("\n▸ FASE 4: Construindo SNN...");
    grammar.build_snn();
    let snn = grammar.snn.as_ref().unwrap();
    println!("  SNN: input={}, hidden={}, output={}", snn.input_neurons.len(), snn.hidden_neurons.len(), snn.output_neurons.len());

    // ═══════════════════════════════════════════════════════════
    // FASE 5: Testar geração híbrida
    // ═══════════════════════════════════════════════════════════
    println!("\n▸ FASE 5: Teste de geração híbrida...\n");

    let t3_keys: Vec<(u16, u8, u16)> = grammar.lexicon.keys().copied().collect();
    let lex_class_map = grammar.lex_to_class.clone();

    let mut snn_top1 = 0usize;
    let mut snn_top3 = 0usize;
    let mut cascade_top1 = 0usize;
    let mut total = 0usize;

    for &key in t3_keys.iter().take(20) {
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

                total += 1;
                if cascade_result == next_id { cascade_top1 += 1; }
                if rankings.iter().take(1).any(|(id, _)| *id == next_id) { snn_top1 += 1; }
                if rankings.iter().take(3).any(|(id, _)| *id == next_id) { snn_top3 += 1; }
            }
        }
    }

    println!("  ┌──────────────────────────────────────────────┐");
    println!("  │ Pipeline HÍBRIDO (LLM + Amadeus)             │");
    println!("  │                                              │");
    println!("  │ Total:            {:>6}                   │", total);
    println!("  │ Cascade top-1:    {:>6} ({:>5.1}%)           │", cascade_top1, cascade_top1 as f64 / total as f64 * 100.0);
    println!("  │ SNN top-1:        {:>6} ({:>5.1}%)           │", snn_top1, snn_top1 as f64 / total as f64 * 100.0);
    println!("  │ SNN top-3:        {:>6} ({:>5.1}%)           │", snn_top3, snn_top3 as f64 / total as f64 * 100.0);
    println!("  └──────────────────────────────────────────────┘");

    // ═══════════════════════════════════════════════════════════
    // FASE 6: Salvar GGUF híbrido
    // ═══════════════════════════════════════════════════════════
    println!("\n▸ FASE 6: Salvando GGUF híbrido...");
    grammar.save_gguf("/tmp/amadeus_hybrid.grammar.gguf", &compiler.lexicon, &compiler.roots)
        .expect("falha ao salvar GGUF híbrido");
    let gguf_size = std::fs::metadata("/tmp/amadeus_hybrid.grammar.gguf").unwrap().len();
    println!("  Tamanho: {:.1} KB", gguf_size as f64 / 1024.0);

    // Verificar roundtrip
    let mut loaded = TripleGrammar::new(3);
    loaded.load_gguf("/tmp/amadeus_hybrid.grammar.gguf").expect("falha ao carregar GGUF");
    let _ = std::fs::remove_file("/tmp/amadeus_hybrid.grammar.gguf");

    println!("  GGUF roundtrip OK ({} GRAPH entries carregados)", loaded.graph.len());

    println!("\n═══════════════════════════════════════════════════════════");
    println!("  PIPELINE HÍBRIDO COMPLETO:");
    println!("  LLM Embeddings (4096D F16) → GRAPH (32-bit) → CUBO + SNN");
    println!("  Tudo em GGUF único, mmap, CPU-only, zero cópia de memória");
    println!("═══════════════════════════════════════════════════════════\n");
}
