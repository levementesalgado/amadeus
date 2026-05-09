use amadeus::amadeus_m::{AmadeusMConfig, AmadeusMModel};
use amadeus::tensor::ops::argmax;
use std::time::Instant;

fn main() {
    println!("╔═╗╔═╗╔═╗╔╦╗╔═╗╦═╗╦╔╗╔╔═╗");
    println!("║╣ ╠═╣║╣  ║ ║╣ ╠╦╝║║║║╚═╗");
    println!("╚═╝╩ ╩╚═╝ ╩ ╚═╝╩╚═╩╝╚╝╚═╝");
    println!("  Amadeus-M (Filosófico)\n");

    let cfg = AmadeusMConfig::small();
    println!("Config:");
    println!("  Layers: {}", cfg.n_layers);
    println!("  Embed: {}", cfg.n_embd);
    println!("  Heads: {} ({} KV heads)", cfg.n_heads, cfg.n_kv_heads);
    println!("  Head dim: {}", cfg.head_dim);
    println!("  Intermediate: {}", cfg.n_intermediate);
    println!("  Vocab: {}", cfg.vocab_size);
    println!("  Window: {}", cfg.window_size);
    println!("  Affect dim: {}", cfg.d_affect);
    println!("  Intent dim: {}", cfg.d_intent);
    println!("  Intent labels: {}", cfg.n_intents);
    println!("  Trace dim: {}", cfg.d_trace);
    println!();

    let start = Instant::now();
    let mut model = AmadeusMModel::new(cfg);
    println!("Model created in {:.2}s", start.elapsed().as_secs_f64());

    let prompt: Vec<u32> = vec![104, 101, 108, 108, 111]; // "hello" in bytes

    println!("\nPrefilling {} tokens...", prompt.len());
    let gen_start = Instant::now();
    for (i, &token) in prompt.iter().enumerate() {
        let _logits = model.forward_token(token, i);
        if i % 1 == 0 {
            print!(".");
        }
    }
    println!(" done");

    let mut next_token = *prompt.last().unwrap();
    let max_new = 20;
    let mut pos = prompt.len();

    println!("\nGenerating {} tokens...", max_new);
    for i in 0..max_new {
        let logits = model.forward_token(next_token, pos);
        let last_dim = logits.len();
        let next = argmax(&logits, last_dim) as u32;
        next_token = next;
        pos += 1;

        if next >= 32 && next < 127 {
            print!("{}", next as u8 as char);
        } else {
            print!("[{next}]");
        }
        std::io::Write::flush(&mut std::io::stdout()).ok();
    }

    println!();
    let elapsed = gen_start.elapsed().as_secs_f64();
    let total = prompt.len() + max_new;
    println!("\nProcessed {total} tokens in {elapsed:.2}s ({:.1} tok/s)",
        total as f64 / elapsed.max(0.001));

    println!("\n--- Estado interno final ---");
    println!("Mood: [{}]", &model.affect.mood.iter()
        .map(|v| format!("{:.3}", v))
        .collect::<Vec<_>>().join(", "));
    println!("Mood intensity: {:.3}", model.affect.intensity());
    println!("Trace (first 8): [{}]", &model.trace[..8.min(model.trace.len())].iter()
        .map(|v| format!("{:.3}", v))
        .collect::<Vec<_>>().join(", "));
}
