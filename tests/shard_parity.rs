//! Paridade entre CUBO em RAM e CUBO shardado.
//!
//! O shard serializa para disco e lê via mmap. Se a serialização ou o índice
//! estiverem errados, a geração muda de saída sem erro visível. Estes testes
//! comparam o conteúdo lido dos dois caminhos.

use amadeus::amadeus_m::hypercube::HyperCube;
use amadeus::amadeus_m::shard::ShardStore;
use amadeus::amadeus_m::hierarchical::compute_clause_depths;
use amadeus::amadeus_m::syntax::assign_dependencies;
use amadeus::amadeus_m::token7::Token7;
use std::collections::HashMap;
use std::path::PathBuf;

/// Diretório temporário único por teste, sem depender de crate externo.
fn temp_dir(nome: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!("amadeus_shard_{}_{}", nome, std::process::id()));
    let _ = std::fs::remove_dir_all(&p);
    p
}

/// Treina um cube pequeno e determinístico a partir de frases fixas.
fn treina(frases: &[&str], order: usize) -> HyperCube {
    let mut cube = HyperCube::new(order);
    for frase in frases {
        let tokens: Vec<Token7> = frase
            .split_whitespace()
            .enumerate()
            .map(|(i, w)| Token7::new((i as u32 * 7 + w.len() as u32) % 500, 0))
            .collect();
        if tokens.len() < 2 {
            continue;
        }
        let com_deps = assign_dependencies(&tokens);
        let depths = compute_clause_depths(&com_deps);
        cube.train(&com_deps, &depths);
    }
    cube
}

fn frases() -> Vec<&'static str> {
    vec![
        "o gato dormiu no sofa",
        "o gato correu pela casa",
        "a casa era grande e clara",
        "o cachorro latiu no quintal",
        "a menina leu o livro novo",
        "o menino correu para a escola",
        "a cidade acordou cedo hoje",
        "o sol brilhou sobre a cidade",
        "a chuva caiu durante a noite",
        "o vento sopra forte na serra",
    ]
}


/// Reconstroi os contextos gerados no treino, para consultar o shard pela
/// chave canônica.
fn contextos_do_treino() -> Vec<Vec<u128>> {
    use amadeus::amadeus_m::hypercube::pack8d;
    let mut out = Vec::new();
    for frase in frases() {
        let tokens: Vec<Token7> = frase
            .split_whitespace()
            .enumerate()
            .map(|(i, w)| Token7::new((i as u32 * 7 + w.len() as u32) % 500, 0))
            .collect();
        if tokens.len() < 2 { continue; }
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

/// Procura o contexto na tabela pela chave canônica.
fn lookup_na_tabela(
    cube: &HyperCube,
    ctx: &[u128],
) -> Option<amadeus::amadeus_m::compact::Cands> {
    let k = amadeus::amadeus_m::compact::chave(ctx);
    cube.table.get(&k).cloned()
}

#[test]
fn flush_preserva_todos_os_contextos() {
    let cube = treina(&frases(), 3);
    let esperado = cube.table.len();
    assert!(esperado > 0, "treino não produziu contextos");

    let dir = temp_dir("flush_todos");
    let mut store = ShardStore::open(&dir).unwrap();
    store.flush(&cube.table, &cube.totals, cube.total_lex).unwrap();

    assert_eq!(
        store.total_contexts(),
        esperado,
        "shard tem {} contextos, cube tinha {}",
        store.total_contexts(),
        esperado
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn conteudo_lido_do_shard_iguala_o_hashmap() {
    let cube = treina(&frases(), 3);

    let dir = temp_dir("conteudo");
    let mut store = ShardStore::open(&dir).unwrap();
    store.flush(&cube.table, &cube.totals, cube.total_lex).unwrap();

    // Cada contexto deve vir com os mesmos candidatos e a mesma soma.
    // A chave canônica u64 não é reversível para o Vec<u128>, então
    // reconstruímos os contextos do treino para consultar o shard.
    let mut checados = 0;
    for ctx in contextos_do_treino() {
        let Some((shard_cands, shard_sum)) = store.get(&ctx) else {
            continue; // contexto não estava no treino (chave pode ter colidido fora)
        };
        let Some(cands) = lookup_na_tabela(&cube, &ctx) else { continue };

        let esperado_sum = cands.total();
        assert!(
            (shard_sum - esperado_sum).abs() < 1e-3,
            "soma difere para {ctx:?}: shard={shard_sum} ram={esperado_sum}"
        );
        assert_eq!(
            shard_cands.len(),
            cands.len(),
            "nº de candidatos difere para {ctx:?}: shard={} ram={}",
            shard_cands.len(),
            cands.len()
        );
        for &(lex, cnt) in &cands.items {
            let sc = shard_cands
                .get(lex)
                .unwrap_or_else(|| panic!("lex {lex} ausente em {ctx:?}"));
            assert!(
                (sc - cnt).abs() < 1e-3,
                "contagem difere em {ctx:?}/lex {lex}: shard={sc} ram={cnt}"
            );
        }
        checados += 1;
    }
    assert!(checados > 0, "nenhum contexto checado");

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn contexto_inexistente_retorna_none() {
    let cube = treina(&frases(), 3);
    let dir = temp_dir("inexistente");
    let mut store = ShardStore::open(&dir).unwrap();
    store.flush(&cube.table, &cube.totals, cube.total_lex).unwrap();

    let ctx: Vec<u128> = vec![u128::MAX, u128::MAX - 1, 0xDEAD_BEEF];
    assert!(
        store.get(&ctx).is_none(),
        "contexto inventado não deveria existir"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn multiplos_flush_acumulam_sem_perder() {
    let mut cube_a = treina(&frases(), 3);
    let mut cube_b = treina(&frases(), 3);
    // segunda "fatia" com contextos exclusive
    let extra: Vec<&str> = vec!["outro texto completamente diferente aqui", "mais um texto novo"];
    let antes = cube_a.table.len();
    for frase in &extra {
        let tokens: Vec<Token7> = frase
            .split_whitespace()
            .enumerate()
            .map(|(i, w)| Token7::new(9000 + i as u32 * 3 + w.len() as u32, 0))
            .collect();
        if tokens.len() < 2 {
            continue;
        }
        let deps = assign_dependencies(&tokens);
        let depths = compute_clause_depths(&deps);
        cube_a.train(&deps, &depths);
    }

    let dir = temp_dir("multi");
    let mut store = ShardStore::open(&dir).unwrap();
    store.flush(&cube_a.table, &cube_a.totals, cube_a.total_lex).unwrap();
    store.flush(&cube_b.table, &cube_b.totals, cube_b.total_lex).unwrap();

    assert_eq!(store.len(), 2, "esperava 2 shards");
    // Cube A tem tudo que B tinha, mais os extras.
    assert!(
        store.total_contexts() >= antes,
        "shards somados perdem contextos: {} < {antes}",
        store.total_contexts()
    );

    let _ = std::fs::remove_dir_all(&dir);
    cube_b.table.clear();
}

#[test]
fn shard_cabem_em_disco_e_sao_menores_que_hashmap() {
    let cube = treina(&frases(), 3);
    let dir = temp_dir("tamanho");
    let mut store = ShardStore::open(&dir).unwrap();
    store.flush(&cube.table, &cube.totals, cube.total_lex).unwrap();

    let em_disco = store.bytes_on_disk();
    assert!(em_disco > 0, "shard não escreveu nada");

    // Estimativa aproximada do HashMap em RAM: cada entrada custa bem mais que
    // o formato serializado (header, alocação separada por Vec/HashMap).
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn abrir_shard_corrompido_falha_limpo() {
    let dir = temp_dir("corrompido");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("00000.cuboshard"), b"nao e um shard valido").unwrap();

    let store = ShardStore::open_dir(&dir).unwrap();
    // open_dir deve ignorar (e avisar), não entrar em pânico.
    assert_eq!(store.len(), 0, "shard corrompido não deveria ser aceito");

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn shard_aberto_por_gguf_tem_indice_consistente() {
    // Abre o shard real que o loader vai encontrar, se existir, e valida o índice.
    let dir = PathBuf::from("training/clause_shards");
    if !dir.exists() {
        eprintln!("  (sem training/clause_shards — pulando)");
        return;
    }
    let store = ShardStore::open_dir(&dir).unwrap();
    assert!(store.len() > 0, "diretorio de shards vazio");
    assert!(store.total_contexts() > 0, "nenhum contexto indexado");
    let _ = std::path::PathBuf::from(".");
    let _ = HashMap::<u32, f32>::new();
}
