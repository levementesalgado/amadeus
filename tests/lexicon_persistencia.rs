//! O léxico precisa sobreviver ao GGUF. Sem as formas de superfície, o modelo
//! carrega mas não pode ser lido: qualquer decodificação seria por ids
//! adivinhados, e a qualidade do texto gerado fica impossível de verificar.

use amadeus::amadeus_m::triple_grammar::TripleGrammar;
use std::collections::HashMap;

#[test]
fn lexico_sobrevive_ao_gguf() {
    let mut lexicon: HashMap<String, (u32, u16, u8, u16)> = HashMap::new();
    for (i, word) in ["o", "futebol", "é", "um", "esporte", "brasileiro", "música"]
        .iter()
        .enumerate()
    {
        lexicon.insert(word.to_string(), (i as u32, 0, 0, (i as u16) * 7));
    }
    let roots: Vec<String> = lexicon.keys().cloned().collect();

    let mut g = TripleGrammar::new(3);
    g.save_gguf("/tmp/amadeus_lex_test.gguf", &lexicon, &roots)
        .expect("save");

    let mut g2 = TripleGrammar::new(3);
    g2.load_gguf("/tmp/amadeus_lex_test.gguf").expect("load");

    assert_eq!(
        g2.lex_forms.len(),
        7,
        "as 7 formas deveriam ter sido persistidas"
    );
    assert_eq!(g2.lex_forms[0], "o");
    assert_eq!(g2.lex_forms[1], "futebol");
    assert_eq!(g2.lex_forms[2], "é");
    assert_eq!(g2.lex_forms[6], "música");
    assert!(g2.lex_forms.iter().all(|f| !f.is_empty()));
}

#[test]
fn forma_e_ordenada_por_id_e_nao_por_hash() {
    // `compiler_lexicon` é um HashMap: a ordem de iteração é arbitrária. Se o
    // save usasse essa ordem, a forma viraria associated ao id errado.
    let mut lexicon: HashMap<String, (u32, u16, u8, u16)> = HashMap::new();
    lexicon.insert("zebra".into(), (0, 0, 0, 0));
    lexicon.insert("abacaxi".into(), (1, 0, 0, 0));
    lexicon.insert("manga".into(), (2, 0, 0, 0));

    let g = TripleGrammar::new(3);
    g.save_gguf("/tmp/amadeus_lex_ord.gguf", &lexicon, &lexicon.keys().cloned().collect::<Vec<_>>())
        .expect("save");

    let mut g2 = TripleGrammar::new(3);
    g2.load_gguf("/tmp/amadeus_lex_ord.gguf").expect("load");
    assert_eq!(g2.lex_forms[0], "zebra");
    assert_eq!(g2.lex_forms[1], "abacaxi");
    assert_eq!(g2.lex_forms[2], "manga");
}

#[test]
fn gguf_antigo_sem_lexico_nao_quebra_o_load() {
    // Compatibilidade: um arquivo gravado antes desta correção não tem a chave.
    // Deve carregar com `lex_forms` vazio em vez de falhar.
    let mut lexicon: HashMap<String, (u32, u16, u8, u16)> = HashMap::new();
    lexicon.insert("palavra".into(), (0, 0, 0, 0));
    let g = TripleGrammar::new(3);
    g.save_gguf("/tmp/amadeus_lex_antigo.gguf", &lexicon, &vec!["palavra".to_string()])
        .expect("save");

    let mut g2 = TripleGrammar::new(3);
    g2.load_gguf("/tmp/amadeus_lex_antigo.gguf").expect("load");
    assert_eq!(g2.lex_forms.len(), 1);
}
