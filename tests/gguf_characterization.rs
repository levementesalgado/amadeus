//! Inspeção do GGUF treinado. Characterization test: confirma o formato,
//! os tensores e as dimensões do artefato real antes de qualquer mudança.

use amadeus::quant::gguf::parse_header;

fn header(path: &str) -> amadeus::quant::gguf::GgufHeader {
    let data = std::fs::read(path).expect("ler gguf");
    parse_header(&data).expect("parse header")
}

#[test]
fn gguf_treinado_tem_magic_e_versao_validos() {
    let h = header("training/wikipedia_grammar.gguf");
    assert_eq!(h.version, 3, "esperava GGUF v3");
    assert_eq!(h.n_tensors, 19, "numero de tensores mudou");
}

#[test]
fn gguf_treinado_tem_os_tensores_esperados() {
    let h = header("training/wikipedia_grammar.gguf");
    let nomes: Vec<&str> = h.tensor_infos.iter().map(|t| t.name.as_str()).collect();

    // Nomes conferidos contra o artefato real. Note o sufixo `.data`, e que
    // T2/T3 ficam sob prefixo minusculo (`t2.`), diferente das structs Rust.
    for esperado in [
        "lexicon.forms",
        "lexicon.roots",
        "graph.data",
        "lex_to_class.data",
        "t2.head.data",
        "t2.gparent.data",
        "t2.class.data",
        "t3.exact.data",
        "t3.css.data",
        "t3.cs.data",
        "t3.cst.data",
        "t3.class.data",
        "cubo.clause.blob",
        "cubo.sentence.blob",
        "cubo.paragraph.blob",
        "cubo.text.blob",
        "snn.synapses_ih",
        "snn.synapses_ho",
        "snn.output_labels",
    ] {
        assert!(
            nomes.contains(&esperado),
            "tensor ausente: {esperado}\npresentes: {nomes:?}"
        );
    }
}

#[test]
fn graph_tem_centena_de_mil_entradas() {
    let h = header("training/wikipedia_grammar.gguf");
    let graph = h
        .tensor_infos
        .iter()
        .find(|t| t.name == "graph.data")
        .expect("graph.data");

    let numel: usize = graph.shape.iter().map(|&d| d as usize).product();
    // Pares (lex_id, assinatura) de u32. 99301 GRAPH => ~198k u32.
    assert!(
        numel >= 190_000,
        "GRAPH pequena demais: {numel} elementos (esperado ~198k para 99301 entradas)"
    );
    assert_eq!(numel % 2, 0, "GRAPH deve ter pares");
}

#[test]
fn lexicon_tem_ordem_de_mil_palavras() {
    let h = header("training/wikipedia_grammar.gguf");
    let lex = h
        .tensor_infos
        .iter()
        .find(|t| t.name == "lexicon.forms")
        .expect("lexicon.forms");

    let rows = lex.shape[0] as usize;
    assert!(rows >= 1000, "léxico pequeno demais: {rows} formas (esperado 1379)");
}

#[test]
fn bin_e_gguf_tem_escala_compativel() {
    // O .bin e o .gguf sao gravados na mesma execucao (train_wikipedia.rs).
    // Divergencia de ordem de grandeza indicaria arquivo desatualizado.
    let bin = std::fs::metadata("training/wikipedia_grammar.bin").expect("bin");
    let gguf = std::fs::metadata("training/wikipedia_grammar.gguf").expect("gguf");

    let ratio = gguf.len() as f64 / bin.len() as f64;
    assert!(
        (0.5..2.0).contains(&ratio),
        "GGUF {} / BIN {} ratio {ratio:.2} sugere arquivo dessincronizado",
        gguf.len(),
        bin.len()
    );
}
