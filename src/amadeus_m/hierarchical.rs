use crate::amadeus_m::token7::{
    Token7, CLASS_PONTUACAO, CLASS_VERBO, CLASS_SUBSTANTIVO, SYN_ROOT,
};
use crate::amadeus_m::hypercube::HyperCube;
use crate::amadeus_m::syntax::assign_dependencies;

pub const ORDER_CLAUSE: usize = 9;
pub const ORDER_SENTENCE: usize = 4;
pub const ORDER_PARAGRAPH: usize = 64;
pub const ORDER_TEXT: usize = 4;

pub struct HierarchicalCubo {
    pub clause: HyperCube,    // order=9 — tokens na oração
    pub sentence: HyperCube,  // order=4 — resumos de orações na sentença
    pub paragraph: HyperCube, // order=64 — resumos de sentenças no parágrafo
    pub text: HyperCube,      // order=4 — resumos de parágrafos
    rng: fastrand::Rng,
}

impl HierarchicalCubo {
    pub fn new() -> Self {
        Self {
            clause: HyperCube::new(ORDER_CLAUSE),
            sentence: HyperCube::new(ORDER_SENTENCE),
            paragraph: HyperCube::new(ORDER_PARAGRAPH),
            text: HyperCube::new(ORDER_TEXT),
            rng: fastrand::Rng::new(),
        }
    }

    // ─── Treino ───

    pub fn train(&mut self, tokens: &[Token7]) {
        if tokens.len() < 3 { return; }
        let parsed = assign_dependencies(tokens);
        let full_depths = compute_clause_depths(&parsed);

        let clause_bounds = find_clause_boundaries(&parsed);
        let clause_ranges = to_ranges(&clause_bounds, parsed.len());
        let sent_bounds = find_sentence_boundaries(&parsed);
        let sent_ranges = to_ranges(&sent_bounds, parsed.len());

        let clause_summaries: Vec<Token7> = clause_ranges.iter()
            .map(|&(a, b)| summarize_unit(&parsed[a..b]))
            .collect();

        for &(a, b) in &clause_ranges {
            if b - a < 1 { continue; }
            let cl = &parsed[a..b];
            let root = summarize_unit(cl);
            let mut seq = vec![root];
            seq.extend_from_slice(cl);
            let mut depths = vec![0u8; seq.len()];
            depths[0] = full_depths.get(a).copied().unwrap_or(0);
            for (j, &d) in full_depths[a..b].iter().enumerate() {
                depths[j + 1] = d;
            }
            self.clause.train(&seq, &depths);
        }

        let clause_to_sent = map_to_super(&clause_ranges, &sent_ranges);

        for si in 0..sent_ranges.len() {
            let c_idx: Vec<usize> = clause_to_sent.iter()
                .enumerate().filter(|(_, s)| **s == si).map(|(c, _)| c).collect();
            if c_idx.is_empty() { continue; }
            let (a, b) = sent_ranges[si];
            let sent_root = summarize_unit(&parsed[a..b]);
            let mut seq = vec![sent_root];
            for &ci in &c_idx { seq.push(clause_summaries[ci]); }
            let depths = full_depths[a..b].to_vec();
            self.sentence.train(&seq, &depths);
        }

        let sent_summaries: Vec<Token7> = sent_ranges.iter()
            .map(|&(a, b)| summarize_unit(&parsed[a..b]))
            .collect();

        let n_sent = sent_ranges.len();
        if n_sent == 0 { return; }
        let para_size = 8.max(n_sent / 5).min(n_sent);
        let para_groups: Vec<Vec<usize>> = (0..n_sent)
            .collect::<Vec<_>>().chunks(para_size)
            .map(|c| c.to_vec()).collect();

        for group in &para_groups {
            if group.is_empty() { continue; }
            let (pa, _) = sent_ranges[group[0]];
            let (_, pb) = sent_ranges[group[group.len()-1]];
            let para_root = summarize_unit(&parsed[pa..pb]);
            let mut seq = vec![para_root];
            for &si in group { seq.push(sent_summaries[si]); }
            let depths = vec![0u8; seq.len()];
            self.paragraph.train(&seq, &depths);
        }

        let para_summaries: Vec<Token7> = para_groups.iter()
            .map(|g| {
                let (a, _) = sent_ranges[g[0]];
                let (_, b) = sent_ranges[g[g.len()-1]];
                summarize_unit(&parsed[a..b])
            }).collect();

        if para_summaries.len() >= 2 {
            let max = 200.min(para_summaries.len());
            for window in para_summaries[..max].windows(ORDER_TEXT + 1) {
                let depths = vec![0u8; window.len()];
                self.text.train(window, &depths);
            }
        }
    }

    // ─── Geração hierárquica top-down ───

    pub fn generate(&mut self, seed: &[Token7], max_len: usize) -> Vec<Token7> {
        let mut out = seed.to_vec();
        if out.is_empty() { out.push(Token7::new(1, 0)); }

        let mut para_summaries: Vec<Token7> = Vec::new();
        let mut sent_summaries: Vec<Token7> = Vec::new();
        let mut clause_summaries: Vec<Token7> = Vec::new();
        let mut clause_tokens: Vec<Token7> = Vec::new();
        let mut clause_depths: Vec<u8> = Vec::new();

        let mut need_clause = true;
        let mut need_sentence = true;
        let mut need_paragraph = true;

        for _ in 0..max_len {
            if need_paragraph {
                let next = if para_summaries.is_empty() {
                    let d = vec![0u8; 1];
                    self.text.sample_lex(&[], &d, 1.0, 0.08, &mut self.rng)
                } else {
                    let start = para_summaries.len().saturating_sub(ORDER_TEXT);
                    let hist = &para_summaries[start..];
                    let d = vec![0u8; hist.len()];
                    self.text.sample_lex(hist, &d, 1.0, 0.08, &mut self.rng)
                };
                if next == 0 && !para_summaries.is_empty() { break; }
                para_summaries.push(Token7::new(next, 0));
                sent_summaries.clear();
                need_paragraph = false;
                need_sentence = true;
            }

            if need_sentence {
                let mut ctx = vec![para_summaries.last().copied().unwrap_or(Token7::new(0, 0))];
                let start = sent_summaries.len().saturating_sub(ORDER_SENTENCE - 1);
                ctx.extend_from_slice(&sent_summaries[start..]);
                let d = vec![0u8; ctx.len()];
                let lex = self.paragraph.sample_lex(&ctx, &d, 1.0, 0.08, &mut self.rng);
                if lex == 0 && !sent_summaries.is_empty() { break; }
                sent_summaries.push(Token7::new(lex, 0));
                clause_summaries.clear();
                need_sentence = false;
                need_clause = true;
            }

            if need_clause {
                let mut ctx = vec![sent_summaries.last().copied().unwrap_or(Token7::new(0, 0))];
                let start = clause_summaries.len().saturating_sub(ORDER_SENTENCE - 1);
                ctx.extend_from_slice(&clause_summaries[start..]);
                let d = vec![0u8; ctx.len()];
                let lex = self.sentence.sample_lex(&ctx, &d, 1.0, 0.08, &mut self.rng);
                if lex == 0 && !clause_summaries.is_empty() { break; }
                let cs = Token7::new(lex, 0);
                clause_summaries.push(cs);
                clause_tokens = vec![cs];
                clause_depths = vec![0u8; 1];
                need_clause = false;
            }

            let lex = self.clause.sample_lex(&clause_tokens, &clause_depths, 1.0, 0.08, &mut self.rng);
            if lex == 0 { break; }

            let new_tok = Token7::new(lex, 0);

            let next_depth = clause_depths.last().copied().unwrap_or(0);
            let depth_after = next_clause_depth(&new_tok, next_depth);
            clause_depths.push(depth_after);

            clause_tokens.push(new_tok);

            if is_clause_end(&new_tok) {
                if let Some(cs) = clause_summaries.last_mut() {
                    *cs = summarize_unit(&clause_tokens);
                }
                out.extend_from_slice(&clause_tokens);
                clause_tokens.clear();
                clause_depths.clear();
                need_clause = true;

                if clause_summaries.len() >= ORDER_SENTENCE {
                    if let Some(s) = sent_summaries.last_mut() {
                        *s = summarize_unit(&clause_summaries);
                    }
                    need_sentence = true;

                    if sent_summaries.len() >= ORDER_PARAGRAPH {
                        if let Some(p) = para_summaries.last_mut() {
                            *p = summarize_unit(&sent_summaries);
                        }
                        need_paragraph = true;
                    }
                }
            }
        }

        if !clause_tokens.is_empty() { out.extend_from_slice(&clause_tokens); }
        out
    }

    // ─── Save / Load ───

    pub fn save(&self, data: &mut Vec<u8>) {
        self.clause.save(data);
        self.sentence.save(data);
        self.paragraph.save(data);
        self.text.save(data);
    }

    pub fn load(&mut self, data: &[u8], off: &mut usize) {
        self.clause.load(data, off);
        self.sentence.load(data, off);
        self.paragraph.load(data, off);
        self.text.load(data, off);
    }
}

// ─── Clause depth ───

pub fn compute_clause_depths(tokens: &[Token7]) -> Vec<u8> {
    let mut depths = vec![0u8; tokens.len()];
    let mut depth = 0u8;
    for i in 0..tokens.len() {
        let t = &tokens[i];
        let cls = t.morph_class();
        let sf = t.syn_func;
        if cls == 4 || (sf == 3 && cls != 5 && cls != 1) {
            if depth < 15 { depth += 1; }
        }
        depths[i] = depth;
        if cls == 5 && depth > 0 {
            depth = depth.saturating_sub(1);
        }
    }
    depths
}

fn next_clause_depth(t: &Token7, prev_depth: u8) -> u8 {
    let cls = t.morph_class();
    let sf = t.syn_func;
    let mut depth = prev_depth;
    if cls == 4 || (sf == 3 && cls != 5 && cls != 1) {
        if depth < 15 { depth += 1; }
    }
    if cls == 5 && depth > 0 {
        depth = depth.saturating_sub(1);
    }
    depth
}

// ─── Utilitários ───

fn find_clause_boundaries(tokens: &[Token7]) -> Vec<usize> {
    let mut bounds = vec![0];
    for i in 1..tokens.len() {
        let prev = &tokens[i-1];
        if prev.morph_class() == CLASS_PONTUACAO && prev.punct >= 1 && prev.punct <= 4 {
            bounds.push(i);
        }
    }
    bounds
}

fn find_sentence_boundaries(tokens: &[Token7]) -> Vec<usize> {
    let mut bounds = vec![0];
    for i in 1..tokens.len() {
        let prev = &tokens[i-1];
        if prev.morph_class() == CLASS_PONTUACAO && (prev.punct == 1 || prev.punct == 7 || prev.punct == 8) {
            bounds.push(i);
        }
    }
    bounds
}

fn to_ranges(boundaries: &[usize], total_len: usize) -> Vec<(usize, usize)> {
    let mut ranges = Vec::new();
    for i in 0..boundaries.len() {
        let start = boundaries[i];
        let end = if i + 1 < boundaries.len() { boundaries[i+1] } else { total_len };
        if end > start { ranges.push((start, end)); }
    }
    ranges
}

fn map_to_super(sub: &[(usize, usize)], super_: &[(usize, usize)]) -> Vec<usize> {
    let mut map = vec![0usize; sub.len()];
    let mut si = 0;
    for ci in 0..sub.len() {
        while si + 1 < super_.len() && sub[ci].0 >= super_[si].1 { si += 1; }
        map[ci] = si;
    }
    map
}

fn summarize_unit(unit: &[Token7]) -> Token7 {
    unit.iter()
        .find(|t| t.syn_func == SYN_ROOT)
        .copied()
        .or_else(|| unit.iter().find(|t| t.morph_class() == CLASS_VERBO).copied())
        .or_else(|| unit.iter().find(|t| t.morph_class() == CLASS_SUBSTANTIVO).copied())
        .unwrap_or_else(|| unit.first().copied().unwrap_or(Token7::new(0, 0)))
}

fn is_clause_end(t: &Token7) -> bool {
    t.morph_class() == CLASS_PONTUACAO && (1..=4).contains(&t.punct)
}
