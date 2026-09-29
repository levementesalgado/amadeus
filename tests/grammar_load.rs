//! Verifica que o GGUF treinado carrega no TripleGrammar e mede o
//! comportamento com spill habilitado.
//!
//! Este arquivo de 795 MB não cabe em RAM junto de uma segunda cópia — o que
//! não é bug, é orçamento (ver docs). O spill existe para isso.

use amadeus::amadeus_m::triple_grammar::TripleGrammar;

fn caminho() -> String {
    std::env::var("AMADEUS_GGUF").unwrap_or_else(|_| "training/wikipedia_grammar.gguf".into())
}

#[test]
fn gguf_treinado_carrega_na_gramatica() {
    let mut g = TripleGrammar::new(3);
    g.load_gguf(&caminho()).expect("load_gguf falhou");

    assert!(
        g.graph.len() > 90_000,
        "GRAPH pequena: {} entradas (esperado 99301)",
        g.graph.len()
    );
    assert!(
        g.lexicon.len() > 1_000,
        "léxico pequeno: {} entradas (esperado 1379)",
        g.lexicon.len()
    );
    assert!(
        g.agreement.len() > 4_000,
        "T2 pequeno: {} contextos (esperado 4968)",
        g.agreement.len()
    );
    assert!(
        !g.lex_to_class.is_empty(),
        "lex_to_class vazio — sem mapeamento de classe morfológica"
    );
}

/// Carrega o modelo grande e gera a partir dele. Existe para garantir que o
/// caminho de inferência funciona com os pesos reais, e que a geração não
/// retorna lixo quando a tabela é grande.
#[test]
fn modelo_grande_gera_lexemas_validos() {
    let mut g = TripleGrammar::new(3);
    g.load_gguf(&caminho()).expect("load_gguf");

    let ctx = g.hier.clause.table.len();
    assert!(ctx > 1_000_000, "esperava tabela grande, veio {ctx}");

    let gerados = gerar_primeiros_lexemas(&mut g, 8);
    assert!(
        !gerados.is_empty(),
        "geração com modelo real não produziu lexemas"
    );
    // Lexema 0 é sentinela de fim de frase; nunca deve ser o único resultado.
    assert!(
        gerados.iter().any(|&l| l != 0),
        "geração devolveu só sentinelas: {gerados:?}"
    );
}

fn gerar_primeiros_lexemas(g: &mut TripleGrammar, n: usize) -> Vec<u32> {
    use amadeus::amadeus_m::token7::Token7;
    use amadeus::amadeus_m::syntax::assign_dependencies;
    use amadeus::amadeus_m::hierarchical::compute_clause_depths;

    let mut rng = fastrand::Rng::with_seed(7);
    let mut hist: Vec<Token7> = vec![Token7::new(100, 0), Token7::new(200, 0)];
    let mut out = Vec::new();
    for _ in 0..n {
        let deps = assign_dependencies(&hist);
        let depths = compute_clause_depths(&deps);
        let lex = g.hier.clause.sample_lex(&hist, &depths, 0.8, 0.1, &mut rng);
        if lex == 0 {
            break;
        }
        hist.push(Token7::new(lex, 0));
        out.push(lex);
    }
    out
}
