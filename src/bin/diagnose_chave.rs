//! Qual campo do Token7 fragmenta o contexto do CUBO?
//!
//! `pack8d` inclui 9 campos no contexto: lex, morph, syn_bin, syn_func,
//! punct, style, clause e graph. O controle n-gram usa só a palavra. Se um
//! campo varia para o mesmo par de palavras, o contexto se fragmenta em
//! múltiplas chaves, cada uma com menos estatística, e a geração degrada.
//!
//! Para cada combinação de campos, mede quantos contextos distintos o
//! corpus produz e quantas ocorrências cada chave tem. Contexto fragmentado
//! tem muitas chaves com contagem 1 — e aí o modelo não tem estatística
//! para escolher, hence a salada.
//!
//! ```bash
//! cargo run --release --bin diagnose_chave
//! ```

use std::collections::HashMap;
use std::fs;
use std::path::Path;

fn main() {
    // Corpus minimo: so para medir fragmentacao, nao precisa ser grande.
    let mut texto = String::new();
    for dir in ["training/wikipedia", "training/narrative"] {
        if Path::new(dir).exists() {
            for e in fs::read_dir(dir).unwrap().flatten() {
                if e.path().extension().map(|x| x == "txt").unwrap_or(false) {
                    if let Ok(c) = fs::read_to_string(e.path()) {
                        if c.len() > 5000 { texto.push_str(&c); texto.push('\n'); }
                    }
                }
            }
        }
    }
    // Primeiras 300 palavras bastam para a razao de fragmentacao.
    let palavras: Vec<&str> = texto.split_whitespace().take(3000).collect();
    println!("medindo com {} palavras\n", palavras.len());

    // Campos: nome -> fn(palavra) -> u128
    // Reproduz pack8d, mas cada campo isolado.
    let lex_de = |p: &str| -> u128 { let h = crc32(p); (h ^ (h >> 16)) as u128 & 0xFFFF };
    let morph_de = |p: &str| -> u128 { if p.ends_with("ção")||p.ends_with("ões") {0} else if p.ends_with("mente") {4} else {0} };
    let syn_de = |p: &str| -> u128 { 0 }; // sem info de posicao no texto solto
    let punct_de = |p: &str| -> u128 { if p.chars().all(|c| c.is_ascii_punctuation()) && !p.is_empty() {1} else {0} };
    let style_de = |p: &str| -> u128 { let h = crc32(p); ((h as u64).wrapping_mul(0x9E3779B9) & 0x3F) as u128 };
    let graph_de = |p: &str| -> u128 { let h = crc32(p); (h & 0xFFFF) as u128 };
    let clause_de = |_p: &str| -> u128 { 0 };

    let campos: Vec<(&str, Box<dyn Fn(&str)->u128>)> = vec![
        ("lex", Box::new(lex_de)),
        ("morph", Box::new(morph_de)),
        ("punct", Box::new(punct_de)),
        ("style", Box::new(style_de)),
        ("graph", Box::new(graph_de)),
        ("clause", Box::new(clause_de)),
        ("syn", Box::new(syn_de)),
    ];

    // Para cada subconjunto relevante, mede fragmentacao do bigrama.
    let cands: Vec<Vec<&str>> = vec![
        vec!["lex"],
        vec!["lex", "morph"],
        vec!["lex", "punct"],
        vec!["lex", "style"],
        vec!["lex", "graph"],
        vec!["lex", "morph", "style", "graph", "clause", "punct", "syn"],
    ];
    let ordem = 2usize;
    for sub in cands {
        let mut chaves: HashMap<Vec<u128>, u32> = HashMap::new();
        for i in ordem..palavras.len() {
            let ctx: Vec<u128> = (i - ordem..i)
                .map(|j| {
                    let p = palavras[j];
                    sub.iter().fold(0u128, |acc, &f| {
                        let v = campos.iter().find(|(n, _)| *n == f).unwrap().1(p);
                        acc.wrapping_mul(131).wrapping_add(v + 1)
                    })
                })
                .collect();
            *chaves.entry(ctx).or_insert(0) += 1;
        }
        let total: u32 = chaves.values().sum();
        let unicos = chaves.len();
        let sincts = chaves.values().filter(|&&c| c == 1).count();
        let p1 = 100.0 * sincts as f64 / unicos as f64;
        let media = total as f64 / unicos as f64;
        let nome = sub.join("+");
        println!("{nome:40} chaves={unicos:6} media={media:6.2} so-1-ocorrencia={sincts:6} ({p1:.0}%)");
    }
}

fn crc32(s: &str) -> u32 {
    let mut h: u32 = 0xFFFF_FFFF;
    for b in s.as_bytes() {
        h ^= *b as u32;
        for _ in 0..8 { h = if h & 1 != 0 { (h >> 1) ^ 0xEDB8_8320 } else { h >> 1 }; }
    }
    !h
}
