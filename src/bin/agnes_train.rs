use amadeus::amadeus_m::{PedagogicalLoop, Agnes};
use std::path::Path;
use std::io::{self, Write};

fn main() {
    println!("╔═══╗  ╔═╗╔═╗╔╗╔╔═╗╔╦╗╦ ╦");
    println!("║ ══╬══╣ ║║ ║║║║║╣  ║ ╚╦╝");
    println!("╠══╗║  ╚═╝╚═╝╝╚╝╚═╝ ╩  ╩ ");
    println!("║  ║║        Agnes AI");
    println!("╚══╝╝  Amadeus — Gramática Tabular de 3 Níveis\n");

    let mut amadeus = PedagogicalLoop::new();

    if Path::new("amadeus.grammar").exists() {
        amadeus.grammar.load("amadeus.grammar").ok();
        println!("  Gramática carregada de amadeus.grammar");
    }
    if Path::new("amadeus.compiler").exists() {
        amadeus.compiler.load_state("amadeus.compiler").ok();
        println!("  Compilador carregado de amadeus.compiler");
    }
    println!();

    amadeus.grammar.syntax_order = 3;
    amadeus.grammar.temperature = 0.8;
    amadeus.temperature = 0.8;
    amadeus.max_new = 60;
    amadeus.grammar.exploration_rate = 0.04;

    let args: Vec<String> = std::env::args().collect();
    let mut input = String::new();
    if args.iter().any(|a| a == "--train-morphology" || a == "-m") {
        println!("  Treinando morfologia...\n");
        amadeus.train_morphology();
    } else if args.iter().any(|a| a == "--auto") {
        let mut agnes = Agnes::new();
        println!("  Modo automático: Agnes treina Amadeus\n");
        loop {
            agnes = Agnes::new();
            println!("═══ Sessão Agnes ═══\n");
            while agnes.should_continue() {
                let question = agnes.pick_question();
                println!("  Agnes: {}\n", question);
                amadeus.auto_episode(&question, 0.5, 'a');
                let response = amadeus.last_response.clone();
                let eval_sat = agnes.evaluate(&question, &response);
                agnes.record(question.to_string(), response.clone(), eval_sat);
                amadeus.reinforce_last(eval_sat);
                let reaction = agnes.pick_reaction(&response, eval_sat);
                println!("  Agnes: {} (sat: {:.0}%)\n", reaction, eval_sat * 100.0);
                agnes.step();
            }
            agnes.report();
            print!("\n  Repetir sessão? (s/N): ");
            io::stdout().flush().ok();
            input.clear();
            io::stdin().read_line(&mut input).ok();
            if input.trim().to_lowercase() != "s" { break; }
        }
    } else if args.iter().any(|a| a == "--manual") {
        println!("  Modo manual\n");
        loop {
            print!("  Você> ");
            io::stdout().flush().ok();
            input.clear();
            io::stdin().read_line(&mut input).ok();
            let text = input.trim();
            if text.is_empty() { continue; }
            match text {
                "sair" | "exit" | "quit" => break,
                "status" => { amadeus.status(); continue; }
                _ => {
                    if let Some(v) = text.strip_prefix("temp ") {
                        if let Ok(t) = v.trim().parse::<f32>() {
                            amadeus.temperature = t.clamp(0.1, 5.0);
                            amadeus.grammar.temperature = t.clamp(0.1, 5.0);
                        }
                        continue;
                    }
                    if let Some(v) = text.strip_prefix("order ") {
                        if let Ok(n) = v.trim().parse::<usize>() {
                            amadeus.grammar.syntax_order = n.clamp(2, 15);
                        }
                        continue;
                    }
                    if let Some(v) = text.strip_prefix("maxlen ") {
                        if let Ok(n) = v.trim().parse::<usize>() {
                            amadeus.max_new = n.clamp(8, 256);
                        }
                        continue;
                    }
                    if let Some(v) = text.strip_prefix("explore ") {
                        if let Ok(r) = v.trim().parse::<f32>() {
                            amadeus.grammar.exploration_rate = r.clamp(0.0, 1.0);
                        }
                        continue;
                    }
                }
            }
            amadeus.auto_episode(text, 0.8, 'a');
            println!();
        }
    } else {
        let mut agnes = Agnes::new();
        println!("  Agnes e Amadeus conversando...\n");
        loop {
            agnes = Agnes::new();
            println!("═══ Conversa ═══\n");
            while agnes.should_continue() {
                let question = agnes.pick_question();
                println!("  Agnes: {}", question);
                amadeus.auto_episode(&question, 0.5, 'a');
                let response = amadeus.last_response.clone();
                let eval_sat = agnes.evaluate(&question, &response);
                agnes.record(question.to_string(), response.clone(), eval_sat);
                amadeus.reinforce_last(eval_sat);
                let reaction = agnes.pick_reaction(&response, eval_sat);
                println!("\n  Agnes: {} (sat: {:.0}%)", reaction, eval_sat * 100.0);
                agnes.step();
                print!("\n  [Enter] ");
                io::stdout().flush().ok();
                input.clear();
                io::stdin().read_line(&mut input).ok();
            }
            agnes.report();
            if let Some(analise) = agnes.evaluate_session() {
                println!("\n  {} Análise:{}", "─", "─");
                for linha in analise.lines() {
                    println!("  {}", linha);
                }
                println!("  {}", "─".repeat(35));
            }
            print!("\n  Nova conversa? (s/N): ");
            io::stdout().flush().ok();
            input.clear();
            io::stdin().read_line(&mut input).ok();
            if input.trim().to_lowercase() != "s" { break; }
        }
    }

    amadeus.grammar.save("amadeus.grammar").ok();
    amadeus.compiler.save_state("amadeus.compiler").ok();
    println!("\n  Modelo salvo em amadeus.grammar + amadeus.compiler");
}
