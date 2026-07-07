use amadeus::amadeus_m::triple_grammar::TripleGrammar;
use amadeus::amadeus_m::token7::Token7;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::time::Instant;

struct SimpleCompiler {
    lexicon: HashMap<String, (u32, u16, u8, u16)>,
    roots: Vec<String>,
}

impl SimpleCompiler {
    fn new() -> Self {
        let mut c = Self {
            lexicon: HashMap::new(),
            roots: vec![String::new()],
        };
        for p in &[",", ".", "!", "?", ";", ":", "—", "-", "\"", "(", ")", "..."] {
            c.lexicon.insert(p.to_string(), (0, 6u16, 6, 0));
        }
        c
    }

    fn compile(&mut self, text: &str) -> Vec<Token7> {
        let mut tokens = Vec::new();
        let words: Vec<&str> = text.split(|c: char| c.is_whitespace() || c == ',' || c == '.' || c == '!' || c == '?' || c == ';' || c == ':').collect();
        for w in words {
            let w = w.trim();
            if w.is_empty() { continue; }
            if let Some(&(id, morph, _, style)) = self.lexicon.get(w) {
                tokens.push(Token7::new(id, morph).with_style(style));
                continue;
            }
            let lower = w.to_lowercase();
            if let Some(&(id, morph, _, style)) = self.lexicon.get(&lower) {
                tokens.push(Token7::new(id, morph).with_style(style));
                continue;
            }
            let id = self.roots.len() as u32;
            self.roots.push(lower.clone());
            let (class, morph) = if lower.ends_with("ção") || lower.ends_with("ões") { (0, 0u16) }
                else if lower.ends_with("mente") { (4, 4u16) }
                else if lower.ends_with("oso") || lower.ends_with("osa") || lower.ends_with("vel") { (2, 2u16) }
                else if lower.ends_with("ar") || lower.ends_with("er") || lower.ends_with("ir") { (1, 1u16) }
                else if lower == "o" || lower == "a" || lower == "os" || lower == "as" { (3, 3u16) }
                else if lower == "de" || lower == "do" || lower == "da" || lower == "em" || lower == "no" || lower == "na" { (5, 5u16) }
                else { (0, 0u16) };
            let style = (id as u16).wrapping_mul(0x9E37) & 0x3F;
            self.lexicon.insert(lower, (id, morph | class as u16, class, style));
            tokens.push(Token7::new(id, morph | class as u16).with_style(style));
        }
        tokens
    }

    fn reverse(&self) -> HashMap<u32, &str> {
        self.lexicon.iter().map(|(w, &(id, _, _, _))| (id, w.as_str())).collect()
    }
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  AMADEUS — BENCHMARK COMPLETO                          ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();

    // ─── FASE 1: Carregar gramática treinada ───
    println!("▸ FASE 1: Carregando gramática Wikipédia...");

    let mut grammar = TripleGrammar::new(3);
    let bin_path = "training/wikipedia_grammar.bin";
    if Path::new(bin_path).exists() {
        grammar.load(bin_path).unwrap();
        println!("  Gramática carregada: {} lex, {} T2, {} GRAPH",
            grammar.lexicon.len(), grammar.agreement.len(), grammar.graph.len());
    } else {
        println!("  ERRO: Gramática não encontrada!");
        return;
    }

    // Construir SNN
    grammar.build_snn();
    println!("  SNN construída");

    let mut compiler = SimpleCompiler::new();

    // ─── FASE 2: Geração de texto ───
    println!();
    println!("▸ FASE 2: Gerando textos...");

    let seeds = vec![
        "o Brasil é um país",
        "a música brasileira tem",
        "o futebol é um esporte",
        "a história do Brasil",
        "a natureza brasileira",
        "a educação é fundamental",
        "a tecnologia avança",
        "a arte brasileira é",
        "a vida no Brasil",
        "o cinema brasileiro",
    ];

    let mut cascade_texts = Vec::new();
    let mut snn_texts = Vec::new();

    for seed in &seeds {
        let seed_tokens = compiler.compile(seed);
        if seed_tokens.is_empty() { continue; }

        // Cascade
        let start = Instant::now();
        let gen_cascade = grammar.generate(&seed_tokens, 20);
        let cascade_time = start.elapsed();

        // SNN
        let start = Instant::now();
        let gen_snn = grammar.generate_with_snn(&seed_tokens, 20);
        let snn_time = start.elapsed();

        let reverse = compiler.reverse();

        let text_cascade: Vec<&str> = gen_cascade.iter()
            .filter(|t| t.lex != 0)
            .filter_map(|t| reverse.get(&t.lex).copied())
            .collect();

        let text_snn: Vec<&str> = gen_snn.iter()
            .filter(|t| t.lex != 0)
            .filter_map(|t| reverse.get(&t.lex).copied())
            .collect();

        cascade_texts.push(text_cascade.join(" "));
        snn_texts.push(text_snn.join(" "));

        println!("  \"{}\"", seed);
        println!("    Cascade ({}µs): {}", cascade_time.as_micros(), text_cascade.join(" "));
        println!("    SNN    ({}µs): {}", snn_time.as_micros(), text_snn.join(" "));
        println!();
    }

    // ─── FASE 3: Análise estatística ───
    println!("▸ FASE 3: Análise estatística...");

    // Distribuição de classes
    let mut class_dist_cascade: HashMap<u8, usize> = HashMap::new();
    let mut class_dist_snn: HashMap<u8, usize> = HashMap::new();
    let mut total_tokens_cascade = 0;
    let mut total_tokens_snn = 0;

    for text in &cascade_texts {
        for word in text.split_whitespace() {
            total_tokens_cascade += 1;
            // Estimar classe por sufixo
            let class = if word.ends_with("ção") || word.ends_with("ões") { 0 }
                else if word.ends_with("mente") { 4 }
                else if word.ends_with("oso") || word.ends_with("osa") { 2 }
                else if word.ends_with("ar") || word.ends_with("er") || word.ends_with("ir") { 1 }
                else if word == "o" || word == "a" || word == "os" || word == "as" { 3 }
                else { 0 };
            *class_dist_cascade.entry(class).or_insert(0) += 1;
        }
    }

    for text in &snn_texts {
        for word in text.split_whitespace() {
            total_tokens_snn += 1;
            let class = if word.ends_with("ção") || word.ends_with("ões") { 0 }
                else if word.ends_with("mente") { 4 }
                else if word.ends_with("oso") || word.ends_with("osa") { 2 }
                else if word.ends_with("ar") || word.ends_with("er") || word.ends_with("ir") { 1 }
                else if word == "o" || word == "a" || word == "os" || word == "as" { 3 }
                else { 0 };
            *class_dist_snn.entry(class).or_insert(0) += 1;
        }
    }

    println!("  Distribuição de classes:");
    println!("  {:<10} {:<15} {:<15}", "Classe", "Cascade", "SNN");
    println!("  {:<10} {:<15} {:<15}", "─".repeat(10), "─".repeat(15), "─".repeat(15));

    let class_names = ["SUBST", "VERBO", "ADJ", "ART", "PREP", "ADVB", "OUT"];
    for class in 0..7u8 {
        let c_cascade = class_dist_cascade.get(&class).unwrap_or(&0);
        let c_snn = class_dist_snn.get(&class).unwrap_or(&0);
        let p_cascade = if total_tokens_cascade > 0 { *c_cascade as f64 / total_tokens_cascade as f64 * 100.0 } else { 0.0 };
        let p_snn = if total_tokens_snn > 0 { *c_snn as f64 / total_tokens_snn as f64 * 100.0 } else { 0.0 };
        let name = class_names.get(class as usize).unwrap_or(&"?");
        println!("  {:<10} {:>5} ({:>5.1}%) {:>5} ({:>5.1}%)",
            name, c_cascade, p_cascade, c_snn, p_snn);
    }

    // Diversidade de vocabulário
    let unique_words_cascade: usize = cascade_texts.iter()
        .flat_map(|t| t.split_whitespace())
        .collect::<std::collections::HashSet<_>>()
        .len();

    let unique_words_snn: usize = snn_texts.iter()
        .flat_map(|t| t.split_whitespace())
        .collect::<std::collections::HashSet<_>>()
        .len();

    println!();
    println!("  Diversidade:");
    println!("    Cascade: {} palavras únicas / {} total", unique_words_cascade, total_tokens_cascade);
    println!("    SNN:     {} palavras únicas / {} total", unique_words_snn, total_tokens_snn);

    // Repetição
    let avg重复_cascade: f64 = cascade_texts.iter()
        .map(|t| {
            let words: Vec<&str> = t.split_whitespace().collect();
            let total = words.len();
            let unique = words.iter().collect::<std::collections::HashSet<_>>().len();
            if total > 0 { (total - unique) as f64 / total as f64 } else { 0.0 }
        })
        .sum::<f64>() / cascade_texts.len() as f64;

    let avg重复_snn: f64 = snn_texts.iter()
        .map(|t| {
            let words: Vec<&str> = t.split_whitespace().collect();
            let total = words.len();
            let unique = words.iter().collect::<std::collections::HashSet<_>>().len();
            if total > 0 { (total - unique) as f64 / total as f64 } else { 0.0 }
        })
        .sum::<f64>() / snn_texts.len() as f64;

    println!();
    println!("  Taxa de repetição:");
    println!("    Cascade: {:.1}%", avg重复_cascade * 100.0);
    println!("    SNN:     {:.1}%", avg重复_snn * 100.0);

    // ─── FASE 4: Benchmark de performance ───
    println!();
    println!("▸ FASE 4: Benchmark de performance...");

    let seed_tokens = compiler.compile("o Brasil é um país grande e diverso");
    let iterations = 100;

    // Cascade
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = grammar.generate(&seed_tokens, 10);
    }
    let cascade_total = start.elapsed();

    // SNN
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = grammar.generate_with_snn(&seed_tokens, 10);
    }
    let snn_total = start.elapsed();

    let tokens_per_iter = 10;
    let total_tokens = iterations * tokens_per_iter;

    println!("  {} iterações ({} tokens total):", iterations, total_tokens);
    println!("    Cascade: {:.2}ms ({:.0} tok/s)", cascade_total.as_secs_f64() * 1000.0,
        total_tokens as f64 / cascade_total.as_secs_f64());
    println!("    SNN:     {:.2}ms ({:.0} tok/s)", snn_total.as_secs_f64() * 1000.0,
        total_tokens as f64 / snn_total.as_secs_f64());
    println!("    Speedup: {:.2}x", cascade_total.as_secs_f64() / snn_total.as_secs_f64());

    println!();
    println!("═══════════════════════════════════════════════════════════");
    println!("  BENCHMARK COMPLETO");
    println!("═══════════════════════════════════════════════════════════");
}
