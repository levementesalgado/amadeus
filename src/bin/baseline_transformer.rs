//! Baseline: transformer decoder, para comparação com o hipercubo.
//!
//! Existe para responder uma pergunta verificável: **o AMADEUS compensa contra
//! um transformer de tamanho equivalente?** Sem um baseline medido, "o
//! hipercubo é mais vantajoso" é impressão, não resultado.
//!
//! Mesma máquina, mesmo vocabulário, mesma tarefa: carregar um modelo e gerar
//! tokens autoregressivamente. Reporta RAM, tok/s e comprimento alcançado.
//!
//! ```bash
//! cargo run --release --bin baseline_transformer -- 1379 500
//! ```
//!
//! Os pesos são aleatórios (`TransformerLayer::new` aloca zerado, e o
//! tokenizer aqui é dummy). Isso é proposital: mede **custo de inferência**,
//! que é a pergunta. Qualidade exigiria treino, e é outro binário.

use amadeus::model::config::ModelConfig;
use amadeus::model::Model;
use std::time::Instant;

fn rss_mb() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmRSS"))
                .and_then(|l| l.split_whitespace().nth(1).and_then(|v| v.parse().ok()))
        })
        .unwrap_or(0)
        / 1024
}

fn argmax(slice: &[f32], n: usize) -> usize {
    let mut best = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for i in 0..n.min(slice.len()) {
        if slice[i] > bv {
            bv = slice[i];
            best = i;
        }
    }
    best
}

/// Monta um modelo em memória, sem arquivo.
fn modelo(vocab: usize, layers: usize, embd: usize, ctx: usize) -> Model {
    let cfg = ModelConfig {
        name: "baseline".into(),
        architecture: "llama".into(),
        n_layers: layers,
        n_heads: 4,
        n_kv_heads: 2,
        n_embd: embd,
        n_intermediate: embd * 2,
        head_dim: embd / 4,
        max_seq_len: ctx,
        vocab_size: vocab,
        norm_eps: 1e-5,
        quant_type: "F32".into(),
        n_gqa: 2,
        rope_theta: 10000.0,
        rope_scaling: None,
    };

    let mut m = Model {
        config: cfg.clone(),
        layers: (0..cfg.n_layers)
            .map(|_| amadeus::layers::TransformerLayer::new(&cfg))
            .collect(),
        tok_embeddings: amadeus::tensor::Tensor::zeros(vec![cfg.vocab_size, cfg.n_embd]),
        output_weight: amadeus::tensor::Tensor::zeros(vec![cfg.vocab_size, cfg.n_embd]),
        norm: amadeus::tensor::Tensor::zeros(vec![cfg.n_embd]),
        norm_eps: cfg.norm_eps,
        kv_caches: (0..cfg.n_layers)
            .map(|_| amadeus::model::KvCache::new(cfg.max_seq_len, cfg.n_kv_heads, cfg.head_dim))
            .collect(),
    };

    // Preenche com valores pequenos em vez de zero: pesos zerados tiram o
    // softmax da degenerescência e inflariam o tempo por curto-circuito.
    let mut s = 0x9E3779B9u32;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        (s as f32 / u32::MAX as f32 - 0.5) * 0.02
    };
    for v in m.tok_embeddings.as_mut_slice() { *v = next(); }
    for v in m.output_weight.as_mut_slice() { *v = next(); }
    for v in m.norm.as_mut_slice() { *v = 1.0; }
    m
}

fn main() {
    let vocab: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(1379);
    let alvo: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(500);
    let layers: usize = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(4);
    let embd: usize = std::env::args().nth(4).and_then(|s| s.parse().ok()).unwrap_or(256);
    let ctx: usize = std::env::args().nth(5).and_then(|s| s.parse().ok()).unwrap_or(512);

    eprintln!("=== baseline transformer (llama-like) ===");
    eprintln!("vocab {vocab} | {layers} layers | embd {embd} | ctx {ctx}");
    eprintln!("rss antes: {} MB", rss_mb());

    let t = Instant::now();
    let mut m = modelo(vocab, layers, embd, ctx);
    let t_build = t.elapsed().as_secs_f64();
    let rss_modelo = rss_mb();

    // Conta parâmetros para comparar com o custo do hipercubo.
    let params: usize = m.tok_embeddings.as_slice().len()
        + m.output_weight.as_slice().len()
        + m.norm.as_slice().len();
    eprintln!("build: {t_build:.2}s | rss: {rss_modelo} MB");
    eprintln!("embedding+saida: {params} f32 = {} MB", params * 4 / 1048576);

    let mut rng = fastrand::Rng::with_seed(7);
    let mut seq: Vec<u32> = vec![(100 % vocab) as u32, (200 % vocab) as u32];
    let mut n = 0usize;

    let t = Instant::now();
    for _ in 0..alvo {
        let pos = seq.len().min(ctx - 1);
        let logits = m.forward_token(seq[seq.len() - 1], pos);
        let next = argmax(logits.as_slice(), vocab) as u32;
        if next == 0 { break; }
        seq.push(next);
        if seq.len() >= ctx { seq.remove(0); }
        n += 1;
    }
    let t_gen = t.elapsed().as_secs_f64();
    let _ = &mut rng;

    eprintln!();
    eprintln!("geracao: {n} tokens em {t_gen:.2}s = {:.1} tok/s ({:.2} ms/token)",
        n as f64 / t_gen.max(0.0001), t_gen * 1000.0 / n.max(1) as f64);
    eprintln!("rss final: {} MB", rss_mb());

    // Repetições para suavizar o ruído de clock.
    let mut melhor = f64::MAX;
    for _ in 0..3 {
        let mut m2 = modelo(vocab, layers, embd, ctx);
        let mut seq: Vec<u32> = vec![100, 200];
        let t = Instant::now();
        for _ in 0..alvo.min(200) {
            let pos = seq.len().min(ctx - 1);
            let logits = m2.forward_token(seq[seq.len() - 1], pos);
            seq.push(argmax(logits.as_slice(), vocab) as u32);
            if seq.len() >= ctx { seq.remove(0); }
        }
        let dt = t.elapsed().as_secs_f64() / alvo.min(200) as f64;
        melhor = melhor.min(dt);
    }
    eprintln!("melhor tempo por token (melhor de 3): {:.2} ms", melhor * 1000.0);
}
