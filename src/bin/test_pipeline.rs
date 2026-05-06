use amadeus::amadeus_m::compiler::Compiler;
use amadeus::amadeus_m::triple_grammar::TripleGrammar;
use amadeus::amadeus_m::token7::Token7;
use amadeus::amadeus_m::syntax::assign_dependencies;

fn main() {
    println!("╔══════════════════════════════════════╗");
    println!("║  PIPELINE AMADEUS (Tabela Tripla)   ║");
    println!("╚══════════════════════════════════════╝\n");

    let mut comp = Compiler::new("underworld");
    let mut grammar = TripleGrammar::new(3);

    let saved_grammar = std::path::Path::new("amadeus.grammar7");
    let saved_compiler = std::path::Path::new("amadeus.compiler");

    if saved_grammar.exists() {
        grammar.load("amadeus.grammar7").ok();
        println!("  Gramática carregada de amadeus.grammar7");
    }
    if saved_compiler.exists() {
        comp.load_state("amadeus.compiler").ok();
        println!("  Compilador carregado de amadeus.compiler");
    }

    println!("  Léxico: {} entradas\n", comp.lexicon.len());
    println!("  CUBO(Oração): {} contextos", grammar.hier.clause.table.len());
    println!("  T2(Concordância): {} contextos", grammar.agreement.len());
    println!("  T3(Léxico): {} contextos\n", grammar.lexicon.len());

    grammar.temperature = 0.9;
    grammar.exploration_rate = 0.04;

    let sementes = [
        "mundo", "tempo", "vida", "mar", "pensamento",
        "Deus", "homem", "Natureza", "alma", "sonho",
    ];

    for seed_text in &sementes {
        let seed_tokens = comp.compile(seed_text);
        let seed = assign_dependencies(&seed_tokens);

        println!("  Semente: {} → {} tokens", seed_text, seed.len());

        let generated = grammar.generate(&seed, 16);
        let output = comp.decompile(&generated);
        println!("  Gerado: {}\n", output);
    }
}
