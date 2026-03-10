use crate::amadeus_m::token7::{
    Token7, CLASS_SUBSTANTIVO, CLASS_VERBO, CLASS_ADJETIVO, CLASS_ARTIGO,
    CLASS_PREPOSICAO, CLASS_PONTUACAO,
    SYN_ROOT, SYN_SUJEITO, SYN_OD, SYN_OI, SYN_ADV, SYN_ADN, SYN_PREP, SYN_PRED,
};

pub fn assign_dependencies(tokens: &[Token7]) -> Vec<Token7> {
    if tokens.is_empty() { return Vec::new(); }

    let mut out: Vec<Token7> = tokens.to_vec();
    let len = out.len();

    let root_idx = find_root(&out);

    for i in 0..len {
        if i == root_idx {
            out[i].syn_off = 0;
            out[i].syn_func = SYN_ROOT;
            continue;
        }
        let cls = out[i].morph_class();
        match cls {
            CLASS_SUBSTANTIVO => assign_noun(&mut out, i, root_idx),
            CLASS_VERBO => assign_verb(&mut out, i, root_idx),
            CLASS_ADJETIVO => assign_adj(&mut out, i, root_idx),
            CLASS_ARTIGO => assign_art(&mut out, i, root_idx),
            CLASS_PREPOSICAO => assign_prep(&mut out, i, root_idx),
            CLASS_PONTUACAO => { out[i].syn_off = 0; out[i].syn_func = 0; }
            _ => assign_other(&mut out, i, root_idx),
        }
    }

    out
}

fn find_root(tokens: &[Token7]) -> usize {
    for (i, t) in tokens.iter().enumerate() {
        if t.morph_class() == CLASS_VERBO { return i; }
    }
    0
}

fn nearest_head(tokens: &[Token7], from: usize, target_class: u8) -> Option<i16> {
    let mut best: Option<(i16, i16)> = None;
    for (j, t) in tokens.iter().enumerate() {
        if t.morph_class() == target_class && j != from {
            let dist = j as i16 - from as i16;
            let abs = dist.abs();
            if best.map_or(true, |(_, b)| abs < b) {
                best = Some((dist, abs));
            }
        }
    }
    best.map(|(d, _)| d)
}

fn assign_noun(out: &mut [Token7], i: usize, root_idx: usize) {
    let has_prep = i > 0 && out[i-1].morph_class() == CLASS_PREPOSICAO;

    if i < root_idx {
        // Left of verb → likely subject
        out[i].syn_off = -(root_idx as i16 - i as i16);
        out[i].syn_func = if has_prep { SYN_OI } else { SYN_SUJEITO };
    } else {
        // Right of verb → likely object
        out[i].syn_off = -(root_idx as i16 - i as i16);
        out[i].syn_func = if has_prep { SYN_OI } else { SYN_OD };
    }
}

fn assign_verb(out: &mut [Token7], i: usize, root_idx: usize) {
    out[i].syn_off = -(root_idx as i16 - i as i16);
    out[i].syn_func = SYN_PRED;
}

fn assign_adj(out: &mut [Token7], i: usize, _root_idx: usize) {
    // Adjunto adnominal: depends on nearest noun
    if let Some(dist) = nearest_head(out, i, CLASS_SUBSTANTIVO) {
        out[i].syn_off = dist;
        out[i].syn_func = SYN_ADN;
    } else if let Some(dist) = nearest_head(out, i, CLASS_VERBO) {
        out[i].syn_off = dist;
        out[i].syn_func = SYN_PRED;
    }
}

fn assign_art(out: &mut [Token7], i: usize, _root_idx: usize) {
    // Artigo depends on the next noun
    let mut head = i + 1;
    while head < out.len() && out[head].morph_class() != CLASS_SUBSTANTIVO {
        head += 1;
    }
    if head < out.len() {
        out[i].syn_off = -(head as i16 - i as i16);
    } else {
        out[i].syn_off = 0;
    }
    out[i].syn_func = SYN_ADN;
}

fn assign_prep(out: &mut [Token7], i: usize, _root_idx: usize) {
    // Preposição depende do próximo substantivo
    let mut head = i + 1;
    while head < out.len() && out[head].morph_class() != CLASS_SUBSTANTIVO {
        head += 1;
    }
    if head < out.len() {
        out[i].syn_off = -(head as i16 - i as i16);
    } else {
        out[i].syn_off = 0;
    }
    out[i].syn_func = SYN_PREP;
}

fn assign_other(out: &mut [Token7], i: usize, root_idx: usize) {
    if let Some(dist) = nearest_head(out, i, CLASS_VERBO) {
        out[i].syn_off = dist;
        out[i].syn_func = SYN_ADV;
    } else {
        out[i].syn_off = -(root_idx as i16 - i as i16);
        out[i].syn_func = SYN_ADV;
    }
}
