//! A chave canônica u64 não pode perder contextos silenciosamente.
//!
//! `pack8d` usa 50 bits por token, então ordem 3 são 150 bits e não cabem
//! num `u64`. Colidimos de propósito com FxHash. Este teste mede se isso
//! custa alguma coisa no corpus real.

use amadeus::amadeus_m::compact::chave;
use amadeus::amadeus_m::hierarchical::compute_clause_depths;
use amadeus::amadeus_m::hypercube::pack8d;
use amadeus::amadeus_m::syntax::assign_dependencies;
use amadeus::amadeus_m::token7::Token7;
use std::collections::{HashMap, HashSet};

/// Gera contextos a partir de frases, como o compilador faz.
fn contextos(frases: &[String]) -> Vec<Vec<u128>> {
    let mut out = Vec::new();
    for frase in frases {
        let tokens: Vec<Token7> = frase
            .split_whitespace()
            .enumerate()
            .map(|(i, w)| Token7::new((i as u32 * 31 + w.len() as u32 * 7) % 997, 0))
            .collect();
        if tokens.len() < 2 {
            continue;
        }
        let deps = assign_dependencies(&tokens);
        let depths = compute_clause_depths(&deps);
        for i in 0..deps.len() {
            for n in 1..=3.min(i) {
                out.push(
                    (i - n..i)
                        .map(|j| pack8d(&deps[j], depths.get(j).copied().unwrap_or(0)))
                        .collect::<Vec<u128>>(),
                );
            }
        }
    }
    out
}

fn frases_variadas() -> Vec<String> {
    let base = [
        "o gato preto dorme no sofá",
        "a menina leu o livro novo",
        "o menino correu para a escola",
        "a cidade acordou cedo",
        "o cachorro latiu forte",
        "a chuva caiu durante a noite",
        "o rio corta a cidade",
        "o sol brilhou sobre o campo",
    ];
    let mut v = Vec::new();
    // Varia o suficiente para gerar centenas de milhares de contextos.
    for i in 0..2000 {
        for f in &base {
            v.push(format!("{f} numero {i}"));
        }
    }
    v
}

#[test]
fn chave_canonica_nao_perde_contextos_distintos() {
    let todos = contextos(&frases_variadas());
    assert!(todos.len() > 100_000, "amostra pequena demais: {}", todos.len());

    // Distintos por valor de sequência.
    let distintos: HashSet<&Vec<u128>> = todos.iter().collect();
    let n_distintos = distintos.len();

    // Distintos por chave canônica.
    let por_hash: HashMap<u64, Vec<&Vec<u128>>> = {
        let mut m: HashMap<u64, Vec<&Vec<u128>>> = HashMap::new();
        for c in distintos {
            m.entry(chave(c)).or_default().push(c);
        }
        m
    };

    let colidindo: Vec<&Vec<&Vec<u128>>> = por_hash.values().filter(|v| v.len() > 1).collect();
    let perdidos: usize = colidindo.iter().map(|v| v.len() - 1).sum();

    assert_eq!(
        por_hash.len(),
        n_distintos,
        "chaves canônicas distintas: {} para {} contextos distintos",
        por_hash.len(),
        n_distintos
    );
    assert_eq!(
        perdidos, 0,
        "{perdidos} contextos perdidos em colisão, em {} chaves com >1",
        colidindo.len()
    );
}

#[test]
fn duplicata_no_dado_nao_e_colisao_de_hash() {
    // O que parece colisão numa medição ingênua é duplicata no dado: o
    // compilador dá id por posição na frase, então a mesma sequência de
    // posições em frases diferentes dá o mesmo contexto.
    let todos = contextos(&frases_variadas());
    let n = todos.len();
    let distintos: HashSet<&Vec<u128>> = todos.iter().collect();

    assert!(
        n > distintos.len(),
        "esperava duplicatas no dado ({} gerados, {} distintos)",
        n,
        distintos.len()
    );
    // E a taxa de duplicata é alta, como esperado.
    let dup = 1.0 - (distintos.len() as f64 / n as f64);
    assert!(
        dup > 0.1,
        "duplicatas baixa demais: {:.1}% — a medição de colisão pode estar confusa",
        dup * 100.0
    );
}

#[test]
fn chave_distingue_contextos_de_ordem_diferente() {
    // Ordem 1, 2 e 3 produzem sequências de tamanhos diferentes e precisam
    // cair em chaves diferentes.
    let c1 = chave(&[pack8d(&Token7::new(42, 0), 0)]);
    let c2 = chave(&[pack8d(&Token7::new(42, 0), 0), pack8d(&Token7::new(7, 0), 0)]);
    let c3 = chave(&[
        pack8d(&Token7::new(42, 0), 0),
        pack8d(&Token7::new(7, 0), 0),
        pack8d(&Token7::new(9, 0), 0),
    ]);
    assert_ne!(c1, c2);
    assert_ne!(c2, c3);
    assert_ne!(c1, c3);
}

#[test]
fn chave_e_deterministica() {
    let ctx = vec![pack8d(&Token7::new(1, 0), 0), pack8d(&Token7::new(2, 3), 1)];
    assert_eq!(chave(&ctx), chave(&ctx), "mesmo contexto, chaves diferentes");
}
