use crate::amadeus_m::token7::{
    Token7, CLASS_PONTUACAO, SYN_PREP,
};

#[derive(Debug, Clone)]
pub enum SynNode {
    Clause {
        kind: ClauseKind,
        children: Vec<SynNode>,
    },
    Phrase {
        function: u8,
        children: Vec<SynNode>,
    },
    TokenLeaf {
        id: u32,
        morph: u16,
        syn_func: u8,
        class: u8,
    },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ClauseKind {
    Main,
    Coordinated,
    Subordinated,
}

impl SynNode {
    pub fn collect_ids(&self) -> Vec<u32> {
        match self {
            SynNode::TokenLeaf { id, .. } => vec![*id],
            SynNode::Phrase { children, .. } | SynNode::Clause { children, .. } => {
                let mut ids = Vec::new();
                for c in children { ids.extend(c.collect_ids()); }
                ids
            }
        }
    }
}

fn syn_name(code: u8) -> &'static str {
    match code {
        0 => "S",
        1 => "OD",
        2 => "OI",
        3 => "ADV",
        4 => "NUC",
        5 => "ADN",
        6 => "PREP",
        7 => "PRED",
        _ => "?",
    }
}

pub fn build_ast(tokens: &[Token7]) -> SynNode {
    let clauses = split_clauses(tokens);
    let children: Vec<SynNode> = clauses.into_iter().map(build_clause).collect();
    SynNode::Clause {
        kind: ClauseKind::Main,
        children,
    }
}

fn split_clauses(tokens: &[Token7]) -> Vec<Vec<Token7>> {
    let mut clauses: Vec<Vec<Token7>> = Vec::new();
    let mut current: Vec<Token7> = Vec::new();
    for t in tokens {
        if t.morph_class() == CLASS_PONTUACAO {
            if !current.is_empty() {
                clauses.push(current);
                current = Vec::new();
            }
            clauses.push(vec![*t]);
        } else {
            current.push(*t);
        }
    }
    if !current.is_empty() { clauses.push(current); }
    clauses
}

fn build_clause(tokens: Vec<Token7>) -> SynNode {
    if tokens.len() == 1 {
        let t = tokens[0];
        if t.morph_class() == CLASS_PONTUACAO {
            return SynNode::Phrase {
                function: t.syn_func,
                children: vec![SynNode::TokenLeaf { id: t.lex, morph: t.morph, syn_func: t.syn_func, class: t.morph_class() }],
            };
        }
    }

    let mut phrases: Vec<SynNode> = Vec::new();
    let mut current_syn: u8 = 0;
    let mut current_tokens: Vec<Token7> = Vec::new();

    for t in &tokens {
        let syn = t.syn_func;
        if syn == SYN_PREP {
            if !current_tokens.is_empty() {
                flush_phrase(&mut phrases, &mut current_syn, &mut current_tokens);
            }
            current_tokens.push(*t);
            current_syn = SYN_PREP;
        } else if syn == current_syn || current_syn == 0 {
            current_tokens.push(*t);
            current_syn = syn;
        } else {
            flush_phrase(&mut phrases, &mut current_syn, &mut current_tokens);
            current_tokens.push(*t);
            current_syn = syn;
        }
    }
    flush_phrase(&mut phrases, &mut current_syn, &mut current_tokens);

    SynNode::Clause {
        kind: ClauseKind::Main,
        children: phrases,
    }
}

fn flush_phrase(phrases: &mut Vec<SynNode>, syn: &mut u8, tokens: &mut Vec<Token7>) {
    if tokens.is_empty() { return; }
    let children: Vec<SynNode> = tokens.drain(..).map(|t| {
        SynNode::TokenLeaf { id: t.lex, morph: t.morph, syn_func: t.syn_func, class: t.morph_class() }
    }).collect();
    phrases.push(SynNode::Phrase {
        function: *syn,
        children,
    });
    *syn = 0;
}

pub fn display_ast(tokens: &[Token7]) -> String {
    let ast = build_ast(tokens);
    let mut out = String::from("─── AST Sintático ───\n");
    out.push_str(&format_tree_simple(&ast, 0));
    out
}

fn format_tree_simple(node: &SynNode, depth: usize) -> String {
    let prefix = "│ ".repeat(depth);
    match node {
        SynNode::Clause { kind, children } => {
            let tag = match kind {
                ClauseKind::Main => "ORACAO PRINCIPAL",
                ClauseKind::Coordinated => "ORACAO COORDENADA",
                ClauseKind::Subordinated => "ORACAO SUBORDINADA",
            };
            let mut out = format!("{}{}\n", prefix, tag);
            for c in children { out.push_str(&format_tree_simple(c, depth + 1)); }
            out
        }
        SynNode::Phrase { function, children } => {
            let name = syn_name(*function);
            if children.is_empty() {
                format!("{}{}\n", prefix, name)
            } else {
                let mut out = format!("{}{}\n", prefix, name);
                for c in children { out.push_str(&format_tree_simple(c, depth + 1)); }
                out
            }
        }
        SynNode::TokenLeaf { id, morph, syn_func, class } => {
            let name = syn_name(*syn_func);
            format!("{}┬── [{}] id={} class={} morph=0x{:04X}\n", prefix, name, id, class, morph)
        }
    }
}
