//! Mede RAM, throughput e comprimento de geração do hipercubo.
//!
//! O comprimento importa: um modelo que trunca em 6 tokens não serve para
//! conversa, por mais rápido que seja.

use amadeus::amadeus_m::hierarchical::compute_clause_depths;
use amadeus::amadeus_m::syntax::assign_dependencies;
use amadeus::amadeus_m::token7::Token7;
use amadeus::amadeus_m::triple_grammar::TripleGrammar;
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

fn main() {
    let path = std::env::args().nth(1).unwrap();
    let alvo: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(500);
    let shard_dir = std::env::var("SHARD_DIR").ok();

    eprintln!("rss antes: {} MB", rss_mb());
    let t = Instant::now();
    let mut g = TripleGrammar::new(3);
    if let Some(d) = &shard_dir {
        g.shard_dir = Some(d.clone());
    }
    if path.ends_with(".gguf") {
        g.load_gguf(&path).expect("load gguf");
    } else {
        g.load(&path).expect("load bin");
    }
    eprintln!("load: {:.1}s | rss apos: {} MB", t.elapsed().as_secs_f64(), rss_mb());
    eprintln!("contextos (RAM): {}", g.hier.clause.table.len());
    if let Some(s) = g.hier.clause.shards.as_ref() {
        eprintln!("contextos (shards): {} | disco: {:.0} MB",
            s.total_contexts(), s.bytes_on_disk() as f64 / 1048576.0);
    }

    // Semente realista, via compilador mínimo.
    let mut comp = Comp::new();
    let sementes = [
        "o Brasil é um país grande",
        "a música brasileira tem",
        "o futebol é um esporte popular",
        "a cidade acordou cedo",
        "o livro estava na mesa",
        "a chuva caiu forte ontem",
        "ele foi para a escola",
        "o rio corta a cidade",
    ];

    println!("\n{:<32} {:>7}  {}", "semente", "saida", "texto");
    let mut total_pedidos = 0usize;
    let mut total_saida = 0usize;
    let mut truncou = 0usize;

    for s in &sementes {
        let tokens = comp.compile(s);
        if tokens.is_empty() { continue; }
        let deps = assign_dependencies(&tokens);
        let saida = g.generate(&deps, alvo);
        let novos = saida.len().saturating_sub(deps.len());
        total_pedidos += alvo;
        total_saida += novos;
        if novos < alvo { truncou += 1; }
        println!("{:<32} {:>7}  {}", s, novos, comp.decompile(&saida));
    }

    let trunc_str = format!("{}/{}", truncou, sementes.len());
    println!("\n{}/{} tokens ({:.0}%) | truncou em {}",
        total_saida, total_pedidos,
        100.0 * total_saida as f64 / total_pedidos as f64, trunc_str);

    // Throughput limpo: uma sequência longa, sem parse.
    let mut rng = fastrand::Rng::with_seed(7);
    let mut hist: Vec<Token7> = vec![Token7::new(100, 0)];
    let t = Instant::now();
    let mut n = 0usize;
    for _ in 0..alvo {
        let deps = assign_dependencies(&hist);
        let d = compute_clause_depths(&deps);
        let lex = g.hier.clause.sample_lex(&hist, &d, 0.8, 0.1, &mut rng);
        if lex == 0 { break; }
        hist.push(Token7::new(lex, 0));
        n += 1;
    }
    let t_gen = t.elapsed().as_secs_f64();
    eprintln!("\nthroughput: {} tokens em {:.2}s = {:.1} tok/s ({:.2} ms/token)",
        n, t_gen, n as f64 / t_gen.max(0.0001), t_gen * 1000.0 / n.max(1) as f64);
    eprintln!("rss final: {} MB", rss_mb());
}

struct Comp { lexicon: HashMapV, roots: Vec<String> }
type HashMapV = std::collections::HashMap<String, (u32, u16, u8, u16)>;
impl Comp {
    fn new() -> Self {
        let mut c = Self { lexicon: HashMapV::new(), roots: vec![String::new()] };
        for p in &[",", ".", "!", "?", ";", ":", "—", "-"] {
            c.lexicon.insert(p.to_string(), (0, 6u16, 6, 0));
        }
        c
    }
    fn compile(&mut self, text: &str) -> Vec<Token7> {
        let mut tokens = Vec::new();
        for w in text.split(|c: char| c.is_whitespace() || ",.!?;:".contains(c)) {
            let w = w.trim();
            if w.is_empty() { continue; }
            if let Some(&(id, morph, _, style)) = self.lexicon.get(w) {
                tokens.push(Token7::new(id, morph).with_style(style)); continue;
            }
            let lower = w.to_lowercase();
            if let Some(&(id, morph, _, style)) = self.lexicon.get(&lower) {
                tokens.push(Token7::new(id, morph).with_style(style)); continue;
            }
            let id = self.roots.len() as u32;
            self.roots.push(lower.clone());
            let (class, morph) = if lower.ends_with("mente") { (4, 4u16) }
                else if lower.ends_with("ar") || lower.ends_with("er") || lower.ends_with("ir") { (1, 1u16) }
                else if lower == "o" || lower == "a" { (3, 3u16) }
                else if lower == "de" || lower == "do" || lower == "da" { (5, 5u16) }
                else { (0, 0u16) };
            let style = (id as u16).wrapping_mul(0x9E37) & 0x3F;
            self.lexicon.insert(lower, (id, morph | class as u16, class, style));
            tokens.push(Token7::new(id, morph | class as u16).with_style(style));
        }
        tokens
    }
    fn decompile(&self, tokens: &[Token7]) -> String {
        let rev: std::collections::HashMap<u32, &str> =
            self.lexicon.iter().map(|(w, &(id, _, _, _))| (id, w.as_str())).collect();
        tokens.iter().map(|t| if t.lex == 0 { "." } else { rev.get(&t.lex).copied().unwrap_or("?") })
            .collect::<Vec<_>>().join(" ")
    }
}
