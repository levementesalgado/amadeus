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


/// Tempo de CPU do processo (user + system), em segundos.
/// Usado para distinguir velocidade de paralelismo: o transformer usa rayon e
/// satura os cores, então tok/s sozinho não diz quanto custa por core.
fn cpu_s() -> f64 {
    let stat = std::fs::read_to_string("/proc/self/stat").unwrap_or_default();
    let c: Vec<&str> = stat.split_whitespace().collect();
    let utime: f64 = c.get(13).and_then(|v| v.parse().ok()).unwrap_or(0.0);
    let stime: f64 = c.get(14).and_then(|v| v.parse().ok()).unwrap_or(0.0);
    (utime + stime) / 100.0
}

/// Núcleos disponíveis, para normalizar.
fn n_cores() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
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
    let comp = Comp::do_grafo(&g);
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
        println!("{:<32} {:>7}  {}", s, novos, comp.decompile(&g, &saida));
    }

    let trunc_str = format!("{}/{}", truncou, sementes.len());
    println!("\n{}/{} tokens ({:.0}%) | truncou em {}",
        total_saida, total_pedidos,
        100.0 * total_saida as f64 / total_pedidos as f64, trunc_str);

    // Throughput limpo: uma sequência longa, sem parse.
    let mut rng = fastrand::Rng::with_seed(7);
    let mut hist: Vec<Token7> = vec![Token7::new(100, 0)];
    let cpu_ini = cpu_s();
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
    let cpu_gen = cpu_s() - cpu_ini;
    eprintln!("\nthroughput: {} tokens em {:.2}s = {:.1} tok/s ({:.2} ms/token)",
        n, t_gen, n as f64 / t_gen.max(0.0001), t_gen * 1000.0 / n.max(1) as f64);
    eprintln!("cpu: {cpu_gen:.2}s de processo | cores: {}", n_cores());
    eprintln!("uso de cpu: {:.0}% | tok/s por core: {:.1}",
        100.0 * cpu_gen / t_gen.max(0.0001),
        n as f64 / t_gen.max(0.0001) / n_cores() as f64);
    eprintln!("rss final: {} MB", rss_mb());
}

/// Compilador sobre o léxico persistido no GGUF.
///
/// A versão anterior montava os ids por ordem de aparição no prompt, o que
/// produzia ids sem relação com os do treino: o modelo recebia contextos
/// errados e a saída era indecodificável. Agora os ids vêm das formas reais.
struct Comp { forma_para_id: HashMapV }
type HashMapV = std::collections::HashMap<String, u32>;
impl Comp {
    fn do_grafo(g: &TripleGrammar) -> Self {
        let mut forma_para_id = HashMapV::new();
        for (id, forma) in g.lex_forms.iter().enumerate() {
            if !forma.is_empty() {
                forma_para_id.entry(forma.to_lowercase()).or_insert(id as u32);
            }
        }
        Self { forma_para_id }
    }
    fn compile(&self, text: &str) -> Vec<Token7> {
        let mut tokens = Vec::new();
        for w in text.split(|c: char| c.is_whitespace() || ",.!?;:".contains(c)) {
            let w = w.trim();
            if w.is_empty() { continue; }
            if let Some(&id) = self.forma_para_id.get(&w.to_lowercase()) {
                tokens.push(Token7::new(id, 0));
            }
        }
        tokens
    }
    fn decompile(&self, g: &TripleGrammar, tokens: &[Token7]) -> String {
        tokens.iter().map(|t| {
            if t.lex == 0 { ".".to_string() }
            else { g.lex_forms.get(t.lex as usize).cloned().unwrap_or_else(|| "?".into()) }
        }).collect::<Vec<_>>().join(" ")
    }
}
