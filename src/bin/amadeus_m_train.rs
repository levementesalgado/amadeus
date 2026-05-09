use amadeus::amadeus_m::PedagogicalLoop;
use std::io::{self, Write};
use std::path::Path;

fn main() {
    println!("╔═╗╔═╗╔═╗╔╦╗╔═╗╦═╗╦╔╗╔╔═╗    ╔═╗╦═╗╔═╗╔═╗╔═╗╔═╗╦  ╦");
    println!("║╣ ╠═╣║╣  ║ ║╣ ╠╦╝║║║║╚═╗    ╠═╣╠╦╝╠═╣╠═╝║╣ ╚═╗║  ║");
    println!("╚═╝╩ ╩╚═╝ ╩ ╚═╝╩╚═╩╝╚╝╚═╝    ╩ ╩╩╚═╩ ╩╩  ╚═╝╚═╝╩═╝╩");
    println!("        Amadeus — Gramática Tabular de 3 Níveis\n");

    let mut amadeus = PedagogicalLoop::new();

    if Path::new("amadeus.grammar7").exists() {
        amadeus.grammar.load("amadeus.grammar7").ok();
        println!("  Gramática carregada de amadeus.grammar7");
    }
    if Path::new("amadeus.compiler").exists() {
        amadeus.compiler.load_state("amadeus.compiler").ok();
        println!("  Compilador carregado de amadeus.compiler");
    }
    println!();

    let args: Vec<String> = std::env::args().collect();
    let mut explicit_iterations: Option<usize> = None;
    if args.len() > 1 {
        let mut i = 1;
        while i < args.len() {
            let arg = &args[i];
            if arg == "--say" || arg == "-s" {
                i += 1;
                if i < args.len() {
                    amadeus.auto_episode(&args[i], 0.8, 'a');
                    println!();
                }
            } else if let Some(text) = arg.strip_prefix("--say=") {
                amadeus.auto_episode(text, 0.8, 'a');
                println!();
            } else if let Some(n) = arg.strip_prefix("--order=") {
                if let Ok(n) = n.parse::<usize>() {
                    amadeus.grammar.syntax_order = n.clamp(2, 6);
                }
            } else if let Some(t) = arg.strip_prefix("--temp=") {
                if let Ok(t) = t.parse::<f32>() {
                    amadeus.grammar.temperature = t.clamp(0.1, 5.0);
                }
            } else if let Some(n) = arg.strip_prefix("--maxlen=") {
                if let Ok(n) = n.parse::<usize>() {
                    amadeus.max_new = n.clamp(8, 256);
                }
            } else if let Some(r) = arg.strip_prefix("--explore=") {
                if let Ok(r) = r.parse::<f32>() {
                    amadeus.grammar.exploration_rate = r.clamp(0.0, 1.0);
                }
            } else if let Some(n) = arg.strip_prefix("--iterations=") {
                if let Ok(n) = n.parse::<usize>() {
                    explicit_iterations = Some(n.clamp(1, 50));
                }
            } else if arg == "--train-morphology" || arg == "-m" {
                match explicit_iterations {
                    Some(n) => amadeus.train_morphology_with_iterations(n),
                    None => amadeus.train_morphology(),
                }
            } else {
                eprintln!("  Ignorando: {arg}");
            }
            i += 1;
        }
        if args.iter().skip(1).any(|a| a.starts_with("--") || a.starts_with("-")) {
            save_all(&mut amadeus);
            return;
        }
    }

    loop {
        print!("Você> ");
        io::stdout().flush().ok();
        let mut input = String::new();
        if io::stdin().read_line(&mut input).unwrap_or(0) == 0 { break; }
        let input = input.trim();
        if input.is_empty() { continue; }
        match input {
            "sair" | "exit" => break,
            "status" => { amadeus.status(); continue; }
            "/porque" => { amadeus.porque(); continue; }
            "/ast" => { amadeus.show_ast(); continue; }
            "/constitution" => { print!("{}", amadeus.constitution.status()); continue; }
            "/friction" => { amadeus.show_friction(); continue; }
            _ => {
                if let Some(v) = input.strip_prefix("temp ") {
                    if let Ok(t) = v.trim().parse::<f32>() {
                        amadeus.temperature = t.clamp(0.1, 5.0);
                        amadeus.grammar.temperature = t.clamp(0.1, 5.0);
                    }
                    continue;
                }
                if let Some(v) = input.strip_prefix("order ") {
                    if let Ok(n) = v.trim().parse::<usize>() {
                        amadeus.grammar.syntax_order = n.clamp(2, 6);
                    }
                    continue;
                }
                if let Some(v) = input.strip_prefix("maxlen ") {
                    if let Ok(n) = v.trim().parse::<usize>() {
                        amadeus.max_new = n.clamp(8, 256);
                    }
                    continue;
                }
                if let Some(v) = input.strip_prefix("explore ") {
                    if let Ok(r) = v.trim().parse::<f32>() {
                        amadeus.grammar.exploration_rate = r.clamp(0.0, 1.0);
                    }
                    continue;
                }
            }
        }
        amadeus.episode(input);
    }

    save_all(&mut amadeus);
    amadeus.status();
}

fn save_all(amadeus: &mut PedagogicalLoop) {
    amadeus.grammar.save("amadeus.grammar7").ok();
    amadeus.compiler.save_state("amadeus.compiler").ok();
    println!("\n  Modelo salvo em amadeus.grammar7 + amadeus.compiler");
}
