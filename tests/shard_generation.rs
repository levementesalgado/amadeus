//! Geração idêntica entre CUBO em RAM e CUBO shardado.
//!
//! Este é o teste que vale: a serialização pode estar correta (o shard tem os
//! mesmos bytes) e ainda assim a leitura durante a geração divergir. Aqui a
//! mesma sequência de tokens, com semente fixa, deve produzir o mesmo texto nos
//! dois modos.

use amadeus::amadeus_m::hypercube::HyperCube;
use amadeus::amadeus_m::hierarchical::compute_clause_depths;
use amadeus::amadeus_m::syntax::assign_dependencies;
use amadeus::amadeus_m::token7::Token7;
use std::path::PathBuf;

fn temp_dir(nome: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!("amadeus_gen_{}_{}", nome, std::process::id()));
    let _ = std::fs::remove_dir_all(&p);
    p
}

fn frases() -> Vec<&'static str> {
    vec![
        "o gato dormiu no sofá",
        "o gato correu pela casa",
        "a casa era grande e clara",
        "o cachorro latiu no quintal",
        "a menina leu o livro novo",
        "o menino correu para a escola",
        "a cidade acordou cedo hoje",
        "o sol brilhou sobre a cidade",
        "a chuva caiu durante a noite",
        "o vento sopra forte na serra",
        "o gato dormiu no sofá again",
        "a menina leu o livro novo aqui",
        "o menino correu para a escola hoje",
        "a cidade acordou cedo de novo",
    ]
}

fn tokens_de(frase: &str, base: u32) -> Vec<Token7> {
    frase
        .split_whitespace()
        .enumerate()
        .map(|(i, w)| Token7::new(base + i as u32 * 13 + (w.len() as u32 * 7) % 211, 0))
        .collect()
}

fn treina(cube: &mut HyperCube, base: u32) {
    for frase in frases() {
        let tokens = tokens_de(frase, base);
        if tokens.len() < 2 {
            continue;
        }
        let deps = assign_dependencies(&tokens);
        let depths = compute_clause_depths(&deps);
        cube.train(&deps, &depths);
    }
}

/// Gera N tokens com semente fixa e devolve os ids de lexema.
fn gera(cube: &mut HyperCube, seed: Vec<Token7>, n: usize, semente: u64) -> Vec<u32> {
    use amadeus::amadeus_m::syntax::assign_dependencies as assign;
    let mut rng = fastrand::Rng::with_seed(semente);
    let mut out = Vec::new();
    let mut hist = seed;

    for i in 0..n {
        let deps = assign(&hist);
        let depths = compute_clause_depths(&deps);
        let lex = cube.sample_lex(&hist, &depths, 0.8, 0.1, &mut rng);
        if lex == 0 {
            break;
        }
        hist.push(Token7::new(lex, 0));
        out.push(lex);
        let _ = i;
    }
    out
}

#[test]
fn geracao_identica_entre_ram_e_sharded() {
    let dir = temp_dir("paridade");

    // Modo RAM
    let mut em_ram = HyperCube::new(3);
    treina(&mut em_ram, 100);

    // Modo shardado: mesmo treino, com spill ligado e flush final
    let mut em_disco = HyperCube::new(3);
    em_disco.enable_spill(&dir).unwrap();
    treina(&mut em_disco, 100);
    em_disco.flush_pending().unwrap();

    assert!(
        !em_ram.table.is_empty(),
        "treino em RAM não produziu tabela"
    );
    assert!(
        em_ram.table.len() > 0,
        "tabela vazia; teste sem valor"
    );

    let seed = tokens_de("o gato", 100);
    let em_ram_ids = gera(&mut em_ram, seed.clone(), 24, 42);
    let em_disco_ids = gera(&mut em_disco, seed.clone(), 24, 42);

    assert!(
        !em_ram_ids.is_empty(),
        "geração em RAM produziu nada; teste sem valor"
    );
    assert_eq!(
        em_ram_ids,
        em_disco_ids,
        "geração diverge:\n  ram   = {em_ram_ids:?}\n  disco = {em_disco_ids:?}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn spill_acumula_ate_o_limite_antes_de_escrever() {
    let dir = temp_dir("acumula");
    let mut cube = HyperCube::new(3);
    cube.enable_spill(&dir).unwrap();
    cube.spill_threshold = 50; // força flush cedo

    treina(&mut cube, 300);

    // Com threshold baixo, deve ter escrito pelo menos um shard.
    let Some(store) = cube.shards.as_ref() else {
        panic!("spill ligado sem store");
    };
    assert!(
        store.len() >= 1,
        "nenhum shard escrito com threshold 50 após treino de {} frases",
        frases().len()
    );

    // E a tabela em RAM deve ter ficado vazia (ou pequena).
    assert!(
        cube.table.is_empty(),
        "spill ligado mas table cresceu: {} entradas",
        cube.table.len()
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn lookup_encontra_contexto_ja_descartado_da_ram() {
    // O ponto do spill: um contexto treinado e descartado da RAM deve
    // continuar encontrável via shard.
    let dir = temp_dir("descartado");
    let mut cube = HyperCube::new(3);
    cube.enable_spill(&dir).unwrap();
    cube.spill_threshold = 10;
    treina(&mut cube, 700);
    cube.flush_pending().unwrap();

    assert!(
        cube.table.is_empty(),
        "precondição: tabela deveria estar vazia"
    );

    let mut achados = 0;
    for ctx in {
        // reconstrói alguns contextos do treino
        let mut v = Vec::new();
        for frase in frases() {
            let tokens = tokens_de(frase, 700);
            let deps = assign_dependencies(&tokens);
            let depths = compute_clause_depths(&deps);
            for i in 0..deps.len() {
                for n in 1..=3.min(i) {
                    v.push(
                        (i - n..i)
                            .map(|j| {
                                let t = &deps[j];
                                amadeus::amadeus_m::hypercube::pack8d(
                                    t,
                                    depths.get(j).copied().unwrap_or(0),
                                )
                            })
                            .collect::<Vec<u128>>(),
                    );
                }
            }
        }
        v
    } {
        if cube.lookup_ctx(&ctx).is_some() {
            achados += 1;
        }
    }

    assert!(
        achados > 0,
        "nenhum contexto do treino encontrado via shard — spill não preservou dados"
    );

    let _ = std::fs::remove_dir_all(&dir);
}
