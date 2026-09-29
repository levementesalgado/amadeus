//! Controle: n-gram de textbook, para isolar a causa da incoerência.
//!
//! O AMADEUS se apresenta como n-gram, mas tem hierarquia, cláusula e
//! modulação por grafo por cima. Isso torna ambíguo o motivo da salada:
//! é o modelo de n-gram que não dá conta, ou a implementação do AMADEUS
//! que atrapalha?
//!
//! Este binário é o controle honesto: interpolação linear de ordem 1..=3
//! com backoff, sem nada de AMADEUS. É o mínimo teórico de um n-gram. Se ele
//! também produzir salada, a culpa é do n-gram (dados). Se ele produzir
//! texto melhor que o AMADEUS, a culpa é da implementação do AMADEUS.
//!
//! Métrica: perplexidade em held-out. Não usa LLM, não usa self-play, não
//! usa nota de opinião. Perplexidade é calculável e comparável.
//!
//! ```bash
//! cargo run --release --bin controle_ngram
//! ```

use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::time::Instant;

type N = usize;

// ─── corpus: mesma montagem do train_wikipedia ───

fn strip_gutenberg(text: &str) -> String {
    for marker in ["*** START OF", "*** START OF THE PROJECT GUTENBERG"] {
        if let Some(i) = text.find(marker) {
            if let Some(j) = text[i..].find("***") {
                return text[i + j + 3..].to_string();
            }
        }
    }
    text.to_string()
}

const PT_UNI: &[&str] = &[" não ", " uma ", " dos ", " das ", " pelo ", " pela ",
    " também ", " então ", " depois ", " porque ", " quando ", " muito ",
    " era ", " ela ", " seu ", " seus "];
const FR: &[&str] = &[" le ", " la ", " les ", " des ", " est ", " une ", " dans ",
    " que ", " pour ", " qui ", " sur ", " avec ", " pas ", " plus ", " son ",
    " ses ", " cette ", " elle ", " nous ", " vous ", " mais ", " tout ",
    " bien ", " aux ", " ont ", " ete ", " tres "];
const IT: &[&str] = &[" il ", " lo ", " gli ", " di ", " che ", " per ", " non ",
    " una ", " con ", " sono ", " questo ", " della ", " nel ", " alla ", " si ",
    " ma ", " come ", " piu ", " anche ", " degli ", " essere "];
const EN: &[&str] = &[" the ", " of ", " and ", " to ", " in ", " is ", " that ",
    " it ", " for ", " with ", " as ", " was ", " on ", " be ", " at ", " by ",
    " this ", " have ", " from ", " or ", " an ", " they ", " which "];
const PT_STOP: &[&str] = &[" de ", " que ", " e ", " a ", " o ", " os ", " as ",
    " um ", " uma ", " para ", " com ", " não ", " se ", " na ", " no ", " por ",
    " mas ", " foi ", " do ", " da ", " dos ", " das "];

fn is_portuguese(text: &str) -> bool {
    if text.is_empty() { return false; }
    let end = text.char_indices().nth(8000).map(|(i, _)| i).unwrap_or(text.len());
    let s = &text[..end];
    let low = s.to_lowercase();
    let n = s.split_whitespace().count().max(1) as f32;
    let pct = |arr: &[&str]| arr.iter().map(|w| low.matches(w).count()).sum::<usize>() as f32 / n;
    let unic = PT_UNI.iter().map(|w| low.matches(w).count()).sum::<usize>();
    if (pct(FR) > 0.03 || pct(IT) > 0.03 || pct(EN) > 0.05) && unic < 20 { return false; }
    unic >= 20 || (pct(PT_STOP) > 0.02
        && s.chars().filter(|c| "ãõáéíóúâêôçà".contains(*c)).count() as f32 / s.len() as f32 > 0.0015)
}

fn carregar() -> (Vec<String>, Vec<String>) {
    let mut pt = Vec::new();
    let mut en = Vec::new();
    for (dir, sink) in [("training/wikipedia", &mut pt), ("training/narrative", &mut en)] {
        if Path::new(dir).exists() {
            for e in fs::read_dir(dir).unwrap().flatten() {
                let p = e.path();
                if p.extension().map(|x| x == "txt").unwrap_or(false) {
                    if let Ok(c) = fs::read_to_string(&p) {
                        let cl = strip_gutenberg(&c);
                        if cl.len() > 500 && is_portuguese(&cl) { sink.push(cl); }
                    }
                }
            }
        }
    }
    (pt, en)
}

fn tokenizar(t: &str, vocab: &mut HashMap<String, u32>) -> Vec<u32> {
    t.split_whitespace()
        .map(|w| {
            let w = w.to_lowercase();
            if let Some(&id) = vocab.get(&w) { id } else {
                let id = vocab.len() as u32;
                vocab.insert(w, id);
                id
            }
        })
        .collect()
}

// ─── o n-gram propriamente dito ───

struct NGram {
    /// tabela[ordem][contexto] -> contagem por próximo token
    tabela: Vec<HashMap<Vec<u32>, HashMap<u32, f64>>>,
    unigram: HashMap<u32, f64>,
    total: f64,
    lambda: Vec<f64>,
}

impl NGram {
    fn new(ordem: usize) -> Self {
        let mut lambda = vec![0.0; ordem];
        let mut acc = 0.0;
        for (i, l) in lambda.iter_mut().enumerate() {
            *l = 1.0 / (i as f64 + 2.0);
            acc += *l;
        }
        for l in lambda.iter_mut() { *l /= acc; }
        Self {
            tabela: (0..ordem).map(|_| HashMap::new()).collect(),
            unigram: HashMap::new(),
            total: 0.0,
            lambda,
        }
    }

    fn treinar(&mut self, seqs: &[Vec<u32>]) {
        for seq in seqs {
            for &t in seq {
                *self.unigram.entry(t).or_insert(0.0) += 1.0;
                self.total += 1.0;
            }
            for i in 1..seq.len() {
                for n in 1..=self.lambda.len() {
                    if i < n { break; }
                    let ctx = &seq[i - n..i];
                    *self.tabela[n - 1]
                        .entry(ctx.to_vec())
                        .or_default()
                        .entry(seq[i])
                        .or_insert(0.0) += 1.0;
                }
            }
        }
    }

    /// Interpolação linear com backoff até unigram.Este é o n-gram
    /// "de verdade" — o que a literatura chama de stupid backoff/interpolação.
    fn p(&self, ctx: &[u32], alvo: u32) -> f64 {
        let mut p = self.lambda.last().copied().unwrap_or(1.0)
            * self.unigram.get(&alvo).copied().unwrap_or(0.0) / self.total.max(1.0);
        for n in (1..self.lambda.len()).rev() {
            if ctx.len() < n { continue; }
            let c = &ctx[ctx.len() - n..];
            if let Some(cnts) = self.tabela[n - 1].get(c) {
                let tot: f64 = cnts.values().sum();
                if let Some(&cnt) = cnts.get(&alvo) {
                    p = p.max(0.0) * 0.0 + self.lambda[n - 1] * (cnt / tot)
                        + (1.0 - self.lambda[n - 1]) * p;
                } else {
                    p = (1.0 - self.lambda[n - 1]) * p;
                }
            }
        }
        p.max(1e-12)
    }

    /// Perplexidade em held-out. Perplexidade = exp(loss médio por token).
    fn perplexidade(&self, seqs: &[Vec<u32>]) -> (f64, usize) {
        let mut ll = 0.0;
        let mut n = 0;
        for seq in seqs {
            for i in 1..seq.len() {
                ll += self.p(&seq[..i], seq[i]).ln();
                n += 1;
            }
        }
        if n == 0 { return (f64::INFINITY, 0); }
        ((-ll / n as f64).exp(), n)
    }
}

fn main() {
    let ordem: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(3);
    let (wikip, narr) = carregar();
    println!("controle n-gram: ordem {ordem}");
    println!("wikipedia: {} docs | narrativos: {} docs", wikip.len(), narr.len());
    if wikip.is_empty() { eprintln!("corpus vazio"); return; }

    let mut vocab: HashMap<String, u32> = HashMap::new();
    let mut inv: Vec<String> = Vec::new();
    let mut toks = |t: &str, v: &mut HashMap<String, u32>, inv: &mut Vec<String>| -> Vec<u32> {
        let s = t.split_whitespace().map(|w| {
            let w = w.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase();
            if w.is_empty() { return 0u32; }
            if let Some(&id) = v.get(&w) { id } else { let id = v.len() as u32; v.insert(w.clone(), id); inv.push(w); id }
        }).filter(|&x| x != 0).collect::<Vec<u32>>();
        s
    };
    let wiki: Vec<Vec<u32>> = wikip.iter().map(|t| toks(t, &mut vocab, &mut inv)).collect();
    let narr_t: Vec<Vec<u32>> = narr.iter().map(|t| toks(t, &mut vocab, &mut inv)).collect();
    println!("vocabulario: {} | tokens wikipedia: {} | tokens narrativa: {}",
        vocab.len(),
        wiki.iter().map(|s| s.len()).sum::<usize>(),
        narr_t.iter().map(|s| s.len()).sum::<usize>());

    // Treina em Wikipédia, mede em narrativa (held-out): texto que o modelo
    // nunca viu, mesmo idioma.
    let t0 = Instant::now();
    let mut m = NGram::new(ordem);
    m.treinar(&wiki);
    let (ppl, n) = m.perplexidade(&narr_t);
    println!("treino: {:.1}s | held-out (narrativa): {} tokens", t0.elapsed().as_secs_f64(), n);
    println!("perplexidade: {ppl:.1}");

    // Gera amostra a partir de sementes, para leitura humana. Candidatos vêm
    // do backoff de ordem mais alta que tiver candidatos; só se não houver
    // nenhum é que se varre o unigram. Greedy por máxima probabilidade.
    println!("\namostras (semente -> continuacao):");
    for semente in ["o brasil é", "a música", "o futebol", "a cidade", "ele foi"] {
        let mut ctx: Vec<u32> = toks(&format!("{semente} "), &mut vocab, &mut inv);
        let mut out: Vec<String> = Vec::new();
        for _ in 0..24 {
            // acha o maior n com candidatos
            let mut melhor = 0u32;
            let mut bp = -1.0f64;
            let mut algum = false;
            for n in (1..m.lambda.len()).rev() {
                if ctx.len() < n { continue; }
                if let Some(cnts) = m.tabela[n - 1].get(&ctx[ctx.len() - n..]) {
                    for (&c, &cnt) in cnts {
                        let p = m.p(&ctx, c);
                        if p > bp { bp = p; melhor = c; algum = true; }
                    }
                    break; // usa a ordem mais alta com candidatos
                }
            }
            if !algum {
                for &c in m.unigram.keys() {
                    let p = m.p(&ctx, c);
                    if p > bp { bp = p; melhor = c; }
                }
            }
            ctx.push(melhor);
            out.push(inv[melhor as usize].clone());
        }
        println!("  {semente:14} -> {}", out.join(" "));
    }
}
