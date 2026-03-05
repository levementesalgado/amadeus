use amadeus::model::loader::load_gguf;
use amadeus::sampler::{Sampler, SamplerKind};
use amadeus::tensor::ops::argmax;
use amadeus::tokenizer::Tokenizer;
use std::time::Instant;

fn main() {
    println!("Amadeus v{} — CPU-first LLM inference engine", amadeus::VERSION);
    println!();

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: amadeus <model.gguf> [prompt]");
        std::process::exit(1);
    }

    let model_path = &args[1];
    let prompt = if args.len() > 2 {
        args[2..].join(" ")
    } else {
        "Hello, how are you?".to_string()
    };

    println!("Loading model: {model_path}...");
    let start = Instant::now();

    let mut model = match load_gguf(model_path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Failed to load model: {e:?}");
            std::process::exit(1);
        }
    };

    println!("  Architecture: {}", model.config.architecture);
    println!("  Parameters: {} layers, {} heads, {} embd",
        model.config.n_layers, model.config.n_heads, model.config.n_embd);
    println!("  Max seq len: {}", model.config.max_seq_len);
    println!("Loaded in {:.2}s", start.elapsed().as_secs_f64());

    // Build a tokenizer from model config (simplified — real one from GGUF metadata)
    let tokenizer = Tokenizer::new(
        vec![],
        vec![],
    );

    // Encode prompt
    println!("\nPrompt: {prompt}");
    // In a real impl: let tokens = tokenizer.encode(&prompt);
    let tokens: Vec<u32> = prompt.bytes().map(|b| b as u32).collect();
    println!("Tokens: {tokens:?}");

    // Prefill: run all prompt tokens
    println!("\nGenerating...");
    let gen_start = Instant::now();
    let mut generated: Vec<u32> = Vec::new();

    for (i, &token) in tokens.iter().enumerate() {
        let logits = model.forward_token(token, i);
        // we don't sample during prefill, just build KV cache
        let _ = logits;
    }

    // Autoregressive generation
    let mut next_token = *tokens.last().unwrap_or(&0);
    let max_new = 128;
    let mut pos = tokens.len();

    for _ in 0..max_new {
        let mut logits = model.forward_token(next_token, pos);

        let last_dim = logits.last_dim();
        let next = argmax(logits.as_slice(), last_dim) as u32;
        // Alternatively use sampler:
        // let next = sampler.sample(logits.as_mut_slice());

        generated.push(next);
        next_token = next;
        pos += 1;

        // Print as we go
        if let Some(s) = tokenizer.vocab.get(next as usize) {
            print!("{s}");
        } else {
            print!("<{next}>");
        }
        std::io::Write::flush(&mut std::io::stdout()).ok();

        if next == tokenizer.eos {
            break;
        }
    }

    println!();
    println!("\nGenerated {} tokens in {:.2}s ({:.1} tok/s)",
        generated.len(),
        gen_start.elapsed().as_secs_f64(),
        generated.len() as f64 / gen_start.elapsed().as_secs_f64().max(0.001));
}
