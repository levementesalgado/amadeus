use amadeus::amadeus_m::compiler::Compiler;

fn main() {
    println!("╔══════════════════════════════════════╗");
    println!("║  COMPILADOR MORFOLÓGICO — AMADEUS    ║");
    println!("╚══════════════════════════════════════╝\n");

    let mut c = Compiler::new("underworld");
    println!("Léxico carregado: {} entradas\n", c.lexicon.len());

    let tests = [
        "O gato preto dorme tranquilamente no tapete",
    ];

    for text in &tests {
        println!("{}", "─".repeat(50));
        println!("ENTRADA: {}", text);
        println!("{}", "─".repeat(50));

        let tokens = c.compile(text);
        println!("\n  # | {:20} | ID   | Morph (hex) | Class", "Token");
        println!("  {}+{}+{}+{}+{}", "─".repeat(3), "─".repeat(22), "─".repeat(6), "─".repeat(13), "─".repeat(5));
        for (i, t) in tokens.iter().enumerate() {
            let form = c.lexicon.iter().find(|(_, val)| val.0 == t.lex)
                .map(|(f, _)| f.clone()).unwrap_or_else(|| format!("[{}]", t.lex));
            println!("  {:2} | {:20} | {:>4} | {:>11} | {:>3}",
                i + 1, form, t.lex, format!("{:04X}", t.morph), t.morph_class());
        }

        let ids: Vec<u32> = tokens.iter().map(|t| t.lex).collect();
        println!("\n  IDs: {:?}", ids);

        let decompiled = c.decompile(&tokens);
        println!("  Descompilado: {}\n", decompiled);
    }
}
