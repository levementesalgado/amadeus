//! Round-trip do formato GGUF: save → load → comparação campo a campo.
//!
//! Convertido de `src/bin/test_gguf.rs`. O binário treinava com PCFG sintético,
//! salvava, recarregava e comparava 20 campos com um contador de erros. Aqui a
//! comparação virou asserção, para que uma regressão de serialização quebre o
//! build em vez de imprimir "3 erros" num terminal que ninguém lê.

use amadeus::amadeus_m::compiler::Compiler;
use amadeus::amadeus_m::pcfg::generate_many;
use amadeus::amadeus_m::token7::Token7;
use amadeus::amadeus_m::triple_grammar::TripleGrammar;
use std::path::PathBuf;

/// Treina uma gramática com dados sintéticos, determinísticos (semente 42).
fn gramática_treinada() -> (TripleGrammar, Compiler) {
    let compiler = Compiler::new("underworld");
    let mut grammar = TripleGrammar::new(3);

    let seqs = generate_many(&compiler, 500, 42);
    for seq in &seqs {
        if seq.len() >= 2 {
            grammar.train(seq);
        }
    }

    let all_tokens: Vec<Token7> = seqs.iter().flat_map(|s| s.iter().copied()).collect();
    let mut c = compiler;
    c.build_graph(&all_tokens, 3);
    grammar.graph = c.graph.clone();
    grammar.graph_alpha = 0.15;

    (grammar, c)
}

fn arquivo(nome: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!("amadeus_rt_{}_{}", nome, std::process::id()));
    let _ = std::fs::remove_file(&p);
    p
}

#[test]
fn gguf_roundtrip_preserva_configuracao() {
    let (original, compiler) = gramática_treinada();
    let path = arquivo("config.gguf");

    original
        .save_gguf(path.to_str().unwrap(), &compiler.lexicon, &compiler.roots)
        .expect("save_gguf");

    let mut loaded = TripleGrammar::new(3);
    loaded.load_gguf(path.to_str().unwrap()).expect("load_gguf");

    assert_eq!(loaded.syntax_order, original.syntax_order, "syntax_order");
    assert!(
        (loaded.temperature - original.temperature).abs() < 1e-6,
        "temperature: {} vs {}",
        loaded.temperature,
        original.temperature
    );
    assert!(
        (loaded.exploration_rate - original.exploration_rate).abs() < 1e-6,
        "exploration_rate"
    );
    assert!(
        (loaded.graph_alpha - original.graph_alpha).abs() < 1e-6,
        "graph_alpha"
    );

    let _ = std::fs::remove_file(&path);
}

#[test]
fn gguf_roundtrip_preserva_cubo() {
    let (original, compiler) = gramática_treinada();
    let path = arquivo("cubo.gguf");

    original
        .save_gguf(path.to_str().unwrap(), &compiler.lexicon, &compiler.roots)
        .expect("save_gguf");

    let mut loaded = TripleGrammar::new(3);
    loaded.load_gguf(path.to_str().unwrap()).expect("load_gguf");

    assert_eq!(
        loaded.hier.clause.table.len(),
        original.hier.clause.table.len(),
        "C_O (clause)"
    );
    assert_eq!(
        loaded.hier.sentence.table.len(),
        original.hier.sentence.table.len(),
        "C_F (sentence)"
    );
    assert_eq!(
        loaded.hier.paragraph.table.len(),
        original.hier.paragraph.table.len(),
        "C_P (paragraph)"
    );
    assert_eq!(
        loaded.hier.text.table.len(),
        original.hier.text.table.len(),
        "C_T (text)"
    );

    let _ = std::fs::remove_file(&path);
}

#[test]
fn gguf_roundtrip_preserva_t2_e_t3() {
    let (original, compiler) = gramática_treinada();
    let path = arquivo("t2t3.gguf");

    original
        .save_gguf(path.to_str().unwrap(), &compiler.lexicon, &compiler.roots)
        .expect("save_gguf");

    let mut loaded = TripleGrammar::new(3);
    loaded.load_gguf(path.to_str().unwrap()).expect("load_gguf");

    assert_eq!(loaded.agreement.len(), original.agreement.len(), "T2 head");
    assert_eq!(
        loaded.agreement_gparent.len(),
        original.agreement_gparent.len(),
        "T2 gparent"
    );
    assert_eq!(loaded.agreement_class.len(), original.agreement_class.len(), "T2 class");

    assert_eq!(loaded.lexicon.len(), original.lexicon.len(), "T3 exact");
    assert_eq!(
        loaded.lexicon_cls_syn_style.len(),
        original.lexicon_cls_syn_style.len(),
        "T3 css"
    );
    assert_eq!(loaded.lexicon_cls_syn.len(), original.lexicon_cls_syn.len(), "T3 cs");
    assert_eq!(
        loaded.lexicon_cls_style.len(),
        original.lexicon_cls_style.len(),
        "T3 cst"
    );
    assert_eq!(loaded.lexicon_class.len(), original.lexicon_class.len(), "T3 class");

    assert_eq!(loaded.graph.len(), original.graph.len(), "GRAPH");
    assert_eq!(
        loaded.lex_to_class.len(),
        original.lex_to_class.len(),
        "lex_to_class"
    );

    let _ = std::fs::remove_file(&path);
}

#[test]
fn gguf_e_bin_preservam_o_mesmo_conteudo() {
    // Dois formatos, mesma execução de treino. Divergência indicaria
    // serialização assimétrica.
    let (original, compiler) = gramática_treinada();
    let gguf = arquivo("paridade.gguf");
    let bin = arquivo("paridade.bin");

    original
        .save_gguf(gguf.to_str().unwrap(), &compiler.lexicon, &compiler.roots)
        .expect("save_gguf");
    original.save(bin.to_str().unwrap()).expect("save bin");

    let mut do_gguf = TripleGrammar::new(3);
    do_gguf.load_gguf(gguf.to_str().unwrap()).expect("load gguf");
    let mut do_bin = TripleGrammar::new(3);
    do_bin.load(bin.to_str().unwrap()).expect("load bin");

    assert_eq!(
        do_gguf.graph.len(),
        do_bin.graph.len(),
        "GRAPH: gguf {} vs bin {}",
        do_gguf.graph.len(),
        do_bin.graph.len()
    );
    assert_eq!(
        do_gguf.lexicon.len(),
        do_bin.lexicon.len(),
        "T3: gguf {} vs bin {}",
        do_gguf.lexicon.len(),
        do_bin.lexicon.len()
    );
    assert_eq!(
        do_gguf.agreement.len(),
        do_bin.agreement.len(),
        "T2: gguf {} vs bin {}",
        do_gguf.agreement.len(),
        do_bin.agreement.len()
    );

    let _ = std::fs::remove_file(&gguf);
    let _ = std::fs::remove_file(&bin);
}
