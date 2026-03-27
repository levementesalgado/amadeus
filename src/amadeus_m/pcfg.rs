use crate::amadeus_m::compiler::Compiler;
use crate::amadeus_m::token7::{
    Token7, morph_from_class,
    CLASS_SUBSTANTIVO, CLASS_VERBO, CLASS_ADJETIVO, CLASS_ARTIGO,
    CLASS_PREPOSICAO, CLASS_PONTUACAO, CLASS_OUTRO,
};
use crate::amadeus_m::syntax::assign_dependencies;

struct Feat {
    gender: u8,
    number: u8,
    person: u8,
    tense: u8,
}

fn random_feat(rng: &mut fastrand::Rng) -> Feat {
    Feat {
        gender: if rng.bool() { 1 } else { 0 },
        number: if rng.bool() { 1 } else { 0 },
        person: rng.u8(0..3),
        tense: rng.u8(0..4),
    }
}

fn pick(compiler: &Compiler, rng: &mut fastrand::Rng, class: u8, f: &Feat) -> u32 {
    let morph = morph_from_class(class, f.gender, f.number, f.person, f.tense);
    let perfect: Vec<u32> = compiler.lexicon.values()
        .filter(|&&(_, m, c, _)| c == class && m == morph)
        .map(|&(id, _, _, _)| id)
        .collect();
    if !perfect.is_empty() {
        return perfect[rng.usize(0..perfect.len())];
    }
    let any: Vec<u32> = compiler.lexicon.values()
        .filter(|&&(_, _, c, _)| c == class)
        .map(|&(id, _, _, _)| id)
        .collect();
    if any.is_empty() { 0 } else { any[rng.usize(0..any.len())] }
}

fn mk_tok(compiler: &Compiler, rng: &mut fastrand::Rng, class: u8, f: &Feat) -> Token7 {
    let lex = pick(compiler, rng, class, f);
    let morph = morph_from_class(class, f.gender, f.number, f.person, f.tense);
    // Style aleatório para popular T3 com variação de registro
    let style = if class == CLASS_PONTUACAO { 0 } else { rng.u8(0..3) as u16 };
    Token7::new(lex, morph).with_style(style)
}

fn gen_np(compiler: &Compiler, rng: &mut fastrand::Rng, subj: &Feat) -> Vec<Token7> {
    let choice = rng.f32();
    let mut v = Vec::new();
    if choice < 0.30 {
        v.push(mk_tok(compiler, rng, CLASS_ARTIGO, subj));
        v.push(mk_tok(compiler, rng, CLASS_SUBSTANTIVO, subj));
        v.push(mk_tok(compiler, rng, CLASS_ADJETIVO, subj));
    } else if choice < 0.70 {
        v.push(mk_tok(compiler, rng, CLASS_ARTIGO, subj));
        v.push(mk_tok(compiler, rng, CLASS_SUBSTANTIVO, subj));
    } else if choice < 0.85 {
        v.push(mk_tok(compiler, rng, CLASS_OUTRO, subj));
    } else {
        v.push(mk_tok(compiler, rng, CLASS_SUBSTANTIVO, subj));
    }
    v
}

fn gen_vp(compiler: &Compiler, rng: &mut fastrand::Rng, subj: &Feat) -> Vec<Token7> {
    let choice = rng.f32();
    let mut v = Vec::new();
    if choice < 0.35 {
        v.push(mk_tok(compiler, rng, CLASS_VERBO, subj));
    } else if choice < 0.55 {
        v.push(mk_tok(compiler, rng, CLASS_VERBO, subj));
        v.push(mk_tok(compiler, rng, CLASS_OUTRO, &Feat { gender: 0, number: 0, person: 0, tense: 0 }));
    } else if choice < 0.75 {
        v.push(mk_tok(compiler, rng, CLASS_VERBO, subj));
        let obj = random_feat(rng);
        v.append(&mut gen_np(compiler, rng, &obj));
    } else {
        v.push(mk_tok(compiler, rng, CLASS_VERBO, subj));
        v.push(mk_tok(compiler, rng, CLASS_PREPOSICAO, &Feat { gender: 0, number: 0, person: 0, tense: 0 }));
        let obj = random_feat(rng);
        v.append(&mut gen_np(compiler, rng, &obj));
    }
    v
}

pub fn generate_sequence(compiler: &Compiler, rng: &mut fastrand::Rng) -> Vec<Token7> {
    let subj = random_feat(rng);
    let choice = rng.f32();
    let mut seq = Vec::new();

    if choice < 0.60 {
        seq.append(&mut gen_np(compiler, rng, &subj));
        seq.append(&mut gen_vp(compiler, rng, &subj));
    } else if choice < 0.80 {
        seq.append(&mut gen_np(compiler, rng, &subj));
    } else {
        let imp = Feat { gender: 0, number: 0, person: 2, tense: 0 };
        seq.append(&mut gen_vp(compiler, rng, &imp));
    }

    let syn = assign_dependencies(&seq);
    let mut result = syn;
    result.push(Token7::new(0, 0).with_punct(1));
    result
}

pub fn generate_many(compiler: &Compiler, n: usize, seed: u64) -> Vec<Vec<Token7>> {
    let mut rng = fastrand::Rng::with_seed(seed);
    let mut sequences = Vec::with_capacity(n);
    for _ in 0..n {
        sequences.push(generate_sequence(compiler, &mut rng));
    }
    sequences
}
