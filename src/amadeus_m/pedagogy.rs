use crate::amadeus_m::teacher::HumanTeacher;
use crate::amadeus_m::triple_grammar::TripleGrammar;
use crate::amadeus_m::compiler::Compiler;
use crate::amadeus_m::consciousness::ConsciousnessState;
use crate::amadeus_m::collatz::CollatzMode;
use crate::amadeus_m::token7::Token7;
use crate::amadeus_m::syntax::assign_dependencies;
use crate::amadeus_m::syntax_ast::display_ast;
use crate::amadeus_m::constitution::{Constitution, default_constitution, ConstraintAction};
use crate::amadeus_m::friction::{analyze_friction, describe_friction};
use crate::amadeus_m::pcfg::generate_many;

pub struct PedagogicalMoment {
    pub phase: String,
    pub input_tokens: Vec<Token7>,
    pub output_tokens: Vec<Token7>,
    pub output_text: String,
    pub satisfaction: f32,
}

#[derive(Clone)]
pub struct DecisionStep {
    pub collatz_n: u64,
    pub mode: CollatzMode,
    pub pos_chosen: u8,
    pub morph_chosen: u16,
    pub id_chosen: u32,
    pub temp_at_step: f32,
    pub explore_at_step: f32,
}

pub struct PedagogicalLoop {
    pub teacher: HumanTeacher,
    pub grammar: TripleGrammar,
    pub compiler: Compiler,
    pub consciousness: ConsciousnessState,
    pub history: Vec<PedagogicalMoment>,
    pub n_episodes: u32,
    pub temperature: f32,
    pub max_new: usize,
    pub last_response: String,
    pub last_trace: Vec<DecisionStep>,
    pub last_input_tokens: Vec<Token7>,
    pub constitution: Constitution,
}

impl PedagogicalLoop {
    pub fn new() -> Self {
        Self {
            teacher: HumanTeacher::new(16),
            grammar: TripleGrammar::new(3),
            compiler: Compiler::new("underworld"),
            consciousness: ConsciousnessState::new(),
            history: Vec::new(),
            n_episodes: 0,
            temperature: 1.0,
            max_new: 32,
            last_response: String::new(),
            last_trace: Vec::new(),
            last_input_tokens: Vec::new(),
            constitution: default_constitution(),
        }
    }

    fn sample_token(&mut self, history: &[Token7]) -> Token7 {
        let explore = self.consciousness.exploration_rate(self.grammar.exploration_rate);
        let old = self.grammar.exploration_rate;
        self.grammar.exploration_rate = explore;
        let mut tok = self.grammar.sample_token(history);

        if let Some(prev) = history.last() {
            let actions = self.constitution.check_rules(prev.morph_class(), tok.morph_class(), prev.morph as u128);
            for action in &actions {
                match action {
                    ConstraintAction::ForcePOS(pos) => {
                        tok = Token7::new(tok.lex, (tok.morph & !0x07) | *pos as u16)
                            .with_style(tok.style);
                    }
                    ConstraintAction::BlockPOS(pos) => {
                        if tok.morph_class() == *pos {
                            let new_pos = if *pos == 0 { 1 } else { 0 };
                            tok = Token7::new(tok.lex, (tok.morph & !0x07) | new_pos as u16)
                                .with_style(tok.style);
                        }
                    }
                    ConstraintAction::WeightMul(_) => {}
                }
            }
        }

        self.grammar.exploration_rate = old;
        tok
    }

    pub fn episode(&mut self, input_text: &str) {
        self.n_episodes += 1;

        let input_tokens = self.compiler.compile(input_text);
        let tokens = assign_dependencies(&input_tokens);
        self.last_input_tokens = tokens.clone();

        println!("\n─── Episódio {} ───", self.n_episodes);
        println!("  Você: {}", input_text);

        self.grammar.train(&tokens);
        self.consciousness.update_focus(input_text.as_bytes());
        self.consciousness.before_generation(0.5, 0.6);
        self.constitution.apply_amendments(&self.consciousness.collatz);

        self.grammar.temperature = (self.temperature * self.consciousness.temperature_modulation()).clamp(0.1, 5.0);

        let mut out_tokens = tokens.clone();
        self.last_trace.clear();
        for _ in 0..self.max_new {
            self.consciousness.collatz.step();
            let collatz_n = self.consciousness.collatz.n;
            let mode = self.consciousness.collatz.mode();
            let temp_at_step = self.grammar.temperature;
            let explore_base = self.grammar.exploration_rate;
            let modulated_explore = self.consciousness.exploration_rate(explore_base);
            let next = self.sample_token(&out_tokens);
            out_tokens.push(next);
            self.last_trace.push(DecisionStep {
                collatz_n, mode,
                pos_chosen: next.morph_class(), morph_chosen: next.morph, id_chosen: next.lex,
                temp_at_step, explore_at_step: modulated_explore,
            });
            if next.lex == 0 { break; }
        }

        let response_text = self.compiler.decompile(&out_tokens[input_tokens.len()..]);
        self.last_response = response_text.clone();

        let fb = self.teacher.auto_react(0.6, 'a');
        let reinforce = (fb.satisfaction - 0.5) * 2.0 * fb.intensity * 1.5;
        for i in 1..out_tokens.len() {
            let delta = reinforce / (out_tokens.len() - 1) as f32;
            self.grammar.reinforce(&out_tokens[i-1..=i], delta);
        }

        self.history.push(PedagogicalMoment {
            phase: "episodio".into(),
            input_tokens: tokens,
            output_tokens: out_tokens,
            output_text: response_text.chars().take(40).collect(),
            satisfaction: fb.satisfaction,
        });

        println!("  Amadeus: {}", response_text);
        println!("  {}", self.consciousness.status_header());
    }

    pub fn auto_episode(&mut self, input_text: &str, satisfaction: f32, affect: char) {
        self.n_episodes += 1;

        let input_tokens = self.compiler.compile(input_text);
        let tokens = assign_dependencies(&input_tokens);
        self.last_input_tokens = tokens.clone();

        self.grammar.train(&tokens);
        self.consciousness.update_focus(input_text.as_bytes());
        self.consciousness.before_generation(0.5, satisfaction);
        self.constitution.apply_amendments(&self.consciousness.collatz);

        self.grammar.temperature = (self.temperature * self.consciousness.temperature_modulation()).clamp(0.1, 5.0);

        let mut out_tokens = tokens.clone();
        self.last_trace.clear();
        for _ in 0..self.max_new {
            self.consciousness.collatz.step();
            let collatz_n = self.consciousness.collatz.n;
            let mode = self.consciousness.collatz.mode();
            let temp_at_step = self.grammar.temperature;
            let explore_base = self.grammar.exploration_rate;
            let modulated_explore = self.consciousness.exploration_rate(explore_base);
            let next = self.sample_token(&out_tokens);
            out_tokens.push(next);
            self.last_trace.push(DecisionStep {
                collatz_n, mode,
                pos_chosen: next.morph_class(), morph_chosen: next.morph, id_chosen: next.lex,
                temp_at_step, explore_at_step: modulated_explore,
            });
            if next.lex == 0 { break; }
        }

        let response_text = self.compiler.decompile(&out_tokens[input_tokens.len()..]);
        self.last_response = response_text.clone();

        let fb = self.teacher.auto_react(satisfaction, affect);
        let reinforce = (fb.satisfaction - 0.5) * 2.0 * fb.intensity * 1.5;
        for i in 1..out_tokens.len() {
            let delta = reinforce / (out_tokens.len() - 1) as f32;
            self.grammar.reinforce(&out_tokens[i-1..=i], delta);
        }

        self.history.push(PedagogicalMoment {
            phase: "auto".into(),
            input_tokens: tokens,
            output_tokens: out_tokens,
            output_text: response_text.chars().take(40).collect(),
            satisfaction: fb.satisfaction,
        });

        println!("  Amadeus: {}", response_text);
        println!("  Satisfação: {:.0}%", fb.satisfaction * 100.0);
        println!("  {}", self.consciousness.status_header());
    }

    pub fn reinforce_last(&mut self, satisfaction: f32) {
        let fb = self.teacher.auto_react(satisfaction, 'a');
        let reinforce = (fb.satisfaction - 0.5) * 2.0 * fb.intensity * 1.5;
        let tokens = self.compiler.compile(&self.last_response);
        let tokens = assign_dependencies(&tokens);
        for i in 1..tokens.len() {
            let delta = reinforce / (tokens.len() - 1) as f32;
            self.grammar.reinforce(&tokens[i-1..=i], delta);
        }
        self.grammar.train(&tokens);
    }

    pub fn train_from_underworld(&mut self, path: &str) -> usize {
        use std::fs;
        use std::path::Path;

        let mut total_lines = 0usize;
        let path = Path::new(path);
        if !path.exists() { return 0; }

        fn walk_dir(ped: &mut PedagogicalLoop, dir: &Path, total: &mut usize) {
            if let Ok(entries) = fs::read_dir(dir) {
                for entry in entries.flatten() {
                    let entry_path = entry.path();
                    if entry_path.is_dir() {
                        walk_dir(ped, &entry_path, total);
                    } else if entry_path.extension().map(|e| e == "md").unwrap_or(false) {
                        if let Ok(content) = fs::read_to_string(&entry_path) {
                            let tokens = ped.compiler.compile(&content);
                            let tokens = assign_dependencies(&tokens);
                            if tokens.len() >= 2 {
                                ped.grammar.train(&tokens);
                                *total += tokens.len();
                            }
                        }
                    }
                }
            }
        }

        walk_dir(self, path, &mut total_lines);
        println!("  Underworld: {} tokens treinados de .md", total_lines);
        total_lines
    }

    pub fn train_morphology(&mut self) {
        self.train_morphology_with_iterations(5)
    }

    pub fn train_morphology_with_iterations(&mut self, iterations: usize) {
        for iter in 0..iterations {
            let n_pcfg = 2000;
            let pcfg_seqs = generate_many(&self.compiler, n_pcfg, 42 + iter as u64 * 137);
            let mut pcfg_total = 0usize;
            for seq in &pcfg_seqs {
                self.grammar.train(seq);
                pcfg_total += seq.len();
            }

            let uw = self.train_from_underworld("underworld");

            println!("  Iteração {}: PCFG {}t / Underworld {}t",
                iter + 1, pcfg_total, uw);
        }

        // Construir embedding GRAPH de todo o underworld
        let all_uw = self.collect_all_tokens("underworld");
        if all_uw.len() > 10 {
            self.compiler.build_graph(&all_uw, 3);
            // Copiar GRAPH para a gramática
            self.grammar.graph = self.compiler.graph.clone();
            self.grammar.graph_alpha = 0.15;
        }

        let lambda_str: String = self.grammar.hier.clause.lambda.iter()
            .enumerate().map(|(i, l)| format!("λ{}={:.4}", i+1, l))
            .collect::<Vec<_>>().join(" ");
        println!("  Gramática final: C_O={}, C_F={}, C_P={}, C_T={}, T2={}, T2+={}, T3={}, CSS={}, CS={}, CST={}, GRAPH={}",
            self.grammar.hier.clause.table.len(),
            self.grammar.hier.sentence.table.len(),
            self.grammar.hier.paragraph.table.len(),
            self.grammar.hier.text.table.len(),
            self.grammar.agreement.len(),
            self.grammar.agreement_gparent.len(),
            self.grammar.lexicon.len(),
            self.grammar.lexicon_cls_syn_style.len(),
            self.grammar.lexicon_cls_syn.len(),
            self.grammar.lexicon_cls_style.len(),
            self.grammar.graph.len());
        println!("  Lambdas C_O: {}", lambda_str);
        println!("  Compilador: {} entradas no léxico", self.compiler.lexicon.len());
    }

    fn collect_all_tokens(&mut self, path: &str) -> Vec<Token7> {
        use std::fs;
        use std::path::Path;
        let mut all = Vec::new();
        let p = Path::new(path);
        if !p.exists() { return all; }
        fn walk(ped: &mut PedagogicalLoop, dir: &Path, out: &mut Vec<Token7>) {
            if let Ok(entries) = fs::read_dir(dir) {
                for entry in entries.flatten() {
                    let ep = entry.path();
                    if ep.is_dir() { walk(ped, &ep, out); }
                    else if ep.extension().map(|e| e == "md").unwrap_or(false) {
                        if let Ok(content) = fs::read_to_string(&ep) {
                            let tokens = ped.compiler.compile(&content);
                            out.extend(tokens);
                        }
                    }
                }
            }
        }
        walk(self, p, &mut all);
        all
    }

    fn find_form(&self, id: u32) -> Option<String> {
        for (form, &(rid, _, _, _)) in &self.compiler.lexicon {
            if rid == id { return Some(form.clone()); }
        }
        let idx = id as usize;
        if idx < self.compiler.roots.len() {
            let r = self.compiler.roots[idx].clone();
            if !r.is_empty() { return Some(r); }
        }
        None
    }

    pub fn porque(&self) {
        println!("\n─── /porque — Rastro de Decisão ───");
        if self.last_trace.is_empty() {
            println!("  (nenhum rastro disponível)");
            return;
        }
        for (i, step) in self.last_trace.iter().enumerate() {
            let modo = match step.mode {
                CollatzMode::Originalist => "ORIG",
                CollatzMode::Vanguardist => "VANG",
            };
            let form = self.find_form(step.id_chosen).unwrap_or_else(|| format!("[{}]", step.id_chosen));
            println!("  [{:2}] n={:>4} {} T={:.2} E={:.2} → {} (pos={}, morph=0x{:04X})",
                i, step.collatz_n, modo, step.temp_at_step, step.explore_at_step,
                form, step.pos_chosen, step.morph_chosen);
        }
        let counts = self.last_trace.iter().fold((0usize, 0usize), |(o, v), s| {
            match s.mode { CollatzMode::Originalist => (o + 1, v), CollatzMode::Vanguardist => (o, v + 1) }
        });
        println!("  Resumo: {} Originalista / {} Vanguardista / {} passos",
            counts.0, counts.1, self.last_trace.len());
        println!("{}", "─".repeat(50));
    }

    pub fn show_ast(&self) {
        if self.last_input_tokens.is_empty() {
            println!("  (nenhum input disponível)");
            return;
        }
        println!("{}", display_ast(&self.last_input_tokens));
    }

    pub fn show_friction(&self) {
        if self.last_trace.is_empty() {
            println!("  (nenhum rastro disponível)");
            return;
        }
        let report = analyze_friction(&self.last_trace);
        println!("{}", describe_friction(&report));
    }

    pub fn status(&self) {
        println!("\n─── Status ───");
        println!("  Episódios: {}", self.n_episodes);
        println!("  Gramática: C_O={}, C_F={}, C_P={}, C_T={}, T2={}, T2+={}, T3={}, CSS={}, CS={}, CST={}",
            self.grammar.hier.clause.table.len(),
            self.grammar.hier.sentence.table.len(),
            self.grammar.hier.paragraph.table.len(),
            self.grammar.hier.text.table.len(),
            self.grammar.agreement.len(),
            self.grammar.agreement_gparent.len(),
            self.grammar.lexicon.len(),
            self.grammar.lexicon_cls_syn_style.len(),
            self.grammar.lexicon_cls_syn.len(),
            self.grammar.lexicon_cls_style.len());
        println!("  Compilador: {} entradas", self.compiler.lexicon.len());
        let lambda_str: String = self.grammar.hier.clause.lambda.iter()
            .enumerate().map(|(i, l)| format!("λ{}={:.3}", i+1, l))
            .collect::<Vec<_>>().join(" ");
        println!("  Lambdas CUBO: {}", lambda_str);
        println!("  Temperatura: {:.1} | Exploração: {:.0}%", self.grammar.temperature, self.grammar.exploration_rate * 100.0);
        println!("  {}", self.consciousness.status_header());
    }
}
