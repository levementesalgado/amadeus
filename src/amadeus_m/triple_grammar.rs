use std::collections::HashMap;
use std::fs;
use crate::amadeus_m::token7::Token7;
use crate::amadeus_m::syntax::assign_dependencies;
use crate::amadeus_m::hierarchical::HierarchicalCubo;

const FORMAT_VERSION: u32 = 4;

pub struct TripleGrammar {
    pub syntax_order: usize,
    // Hipercubo hierárquico 4 níveis
    pub hier: HierarchicalCubo,
    // T2 — propagação de traços via dependência (multi-hop)
    pub agreement: HashMap<(u16, u16, u8), HashMap<(u16, u16), f32>>,
    pub agreement_totals: HashMap<(u16, u16, u8), f32>,
    pub agreement_class: HashMap<u8, HashMap<(u16, u16), f32>>,
    pub agreement_class_totals: HashMap<u8, f32>,
    pub agreement_gparent: HashMap<(u16, u16, u8), HashMap<(u16, u16), f32>>,
    pub agreement_gparent_totals: HashMap<(u16, u16, u8), f32>,
    // T3 — seleção lexical
    pub lexicon: HashMap<(u16, u8, u16), HashMap<u32, f32>>,
    pub lexicon_totals: HashMap<(u16, u8, u16), f32>,
    pub lexicon_cls_syn_style: HashMap<(u8, u8, u16), HashMap<u32, f32>>,
    pub lexicon_cls_syn_style_totals: HashMap<(u8, u8, u16), f32>,
    pub lexicon_cls_syn: HashMap<(u8, u8), HashMap<u32, f32>>,
    pub lexicon_cls_syn_totals: HashMap<(u8, u8), f32>,
    pub lexicon_cls_style: HashMap<(u8, u16), HashMap<u32, f32>>,
    pub lexicon_cls_style_totals: HashMap<(u8, u16), f32>,
    pub lexicon_class: HashMap<u8, HashMap<u32, f32>>,
    pub lexicon_class_totals: HashMap<u8, f32>,
    // Mapa lex → classe morfológica (populado durante treino)
    pub lex_to_class: HashMap<u32, u8>,
    // Mapa lex → GRAPH embedding (random indexing, populado pelo compiler)
    pub graph: HashMap<u32, u32>,
    pub graph_alpha: f32,
    // Pilha de cláusulas
    pub clause_stack: Vec<u8>,
    // Params
    pub temperature: f32,
    pub exploration_rate: f32,
    pub rng: fastrand::Rng,
    // SNN para T3
    pub snn: Option<crate::amadeus_m::snn::SpikingNetwork>,
    pub snn_enc: Option<crate::amadeus_m::snn::MorphEncoding>,
}

impl TripleGrammar {
    pub fn new(_syntax_order: usize) -> Self {
        Self {
            syntax_order: _syntax_order,
            hier: HierarchicalCubo::new(),
            agreement: HashMap::new(),
            agreement_totals: HashMap::new(),
            agreement_class: HashMap::new(),
            agreement_class_totals: HashMap::new(),
            agreement_gparent: HashMap::new(),
            agreement_gparent_totals: HashMap::new(),
            lexicon: HashMap::new(),
            lexicon_totals: HashMap::new(),
            lexicon_cls_syn_style: HashMap::new(),
            lexicon_cls_syn_style_totals: HashMap::new(),
            lexicon_cls_syn: HashMap::new(),
            lexicon_cls_syn_totals: HashMap::new(),
            lexicon_cls_style: HashMap::new(),
            lexicon_cls_style_totals: HashMap::new(),
            lexicon_class: HashMap::new(),
            lexicon_class_totals: HashMap::new(),
            lex_to_class: HashMap::new(),
            graph: HashMap::new(),
            graph_alpha: 0.15,
            clause_stack: Vec::new(),
            temperature: 1.0,
            exploration_rate: 0.08,
            rng: fastrand::Rng::new(),
            snn: None,
            snn_enc: None,
        }
    }

    // ─── Construir SNN a partir das tabelas gramaticais ───

    pub fn build_snn(&mut self) {
        let (net, enc) = crate::amadeus_m::snn::build_snn_from_grammar(
            &self.lexicon,
            &self.lexicon_cls_syn_style,
            &self.lexicon_cls_syn,
            &self.lexicon_cls_style,
            &self.lexicon_class,
            5.0,    // tau
            1.0,    // threshold
            2,      // refrac
            1.0,    // dt
            16,     // tsteps
        );
        self.snn = Some(net);
        self.snn_enc = Some(enc);
    }

    // ─── Amostragem via SNN (substitui cascade T3) ───

    pub fn snn_sample(&mut self, morph: u16, syn_func: u8, style: u16, candidate: u32) -> u32 {
        let net = self.snn.as_mut().unwrap();
        let enc = self.snn_enc.as_ref().unwrap();
        crate::amadeus_m::snn::snn_sample_lex(
            net, enc, morph, syn_func, style,
            candidate, self.temperature, self.exploration_rate, &mut self.rng,
        )
    }

    // ─── Amostragem com fallback: SNN → cascade legada ───

    pub fn sample_lex_with_snn(&mut self, morph: u16, syn_func: u8, style: u16, cls: u8, candidate: u32) -> u32 {
        if self.snn.is_some() {
            return self.snn_sample(morph, syn_func, style, candidate);
        }
        self.sample_lex_full(morph, syn_func, style, cls, candidate)
    }

    // ─── Rastreio de profundidade de cláusula ───
    // Heurística: PREP/ADV com syn_func de subordinação → push,
    // PONT final → pop. Durante treino, computamos do golden.

    pub fn compute_clause_depths(tokens: &[Token7]) -> Vec<u8> {
        let mut depths = vec![0u8; tokens.len()];
        let mut stack: Vec<usize> = Vec::new();
        let mut depth = 0u8;
        for i in 0..tokens.len() {
            let t = &tokens[i];
            let cls = t.morph_class();
            let sf = t.syn_func;

            // Subordinadores: PREP(4) com syn_func ADV(3) ou PREP(6)
            if cls == 4 || (sf == 3 && cls != 5 && cls != 1) {
                if depth < 15 {
                    depth += 1;
                    stack.push(i);
                }
            }

            depths[i] = depth;

            // Fim de subordinada: PONT(5) ou verbo principal retomando
            if cls == 5 && depth > 0 {
                if let Some(_) = stack.last() {
                    depth = depth.saturating_sub(1);
                    stack.pop();
                }
            }
        }
        depths
    }

    pub fn train(&mut self, tokens: &[Token7]) {
        for i in 0..tokens.len() {
            let t = &tokens[i];
            let cls = t.morph_class();

            // Mapear lex → classe
            self.lex_to_class.entry(t.lex).or_insert(cls);

            // Copiar GRAPH dos tokens compilados
            if t.graph != 0 {
                self.graph.entry(t.lex).or_insert(t.graph);
            }

            // T3 — cascata completa
            let t3key = (t.morph, t.syn_func, t.style);
            *self.lexicon.entry(t3key).or_default().entry(t.lex).or_insert(0.0) += 1.0;
            *self.lexicon_totals.entry(t3key).or_insert(0.0) += 1.0;
            let css = (cls, t.syn_func, t.style);
            *self.lexicon_cls_syn_style.entry(css).or_default().entry(t.lex).or_insert(0.0) += 1.0;
            *self.lexicon_cls_syn_style_totals.entry(css).or_insert(0.0) += 1.0;
            let cs = (cls, t.syn_func);
            *self.lexicon_cls_syn.entry(cs).or_default().entry(t.lex).or_insert(0.0) += 1.0;
            *self.lexicon_cls_syn_totals.entry(cs).or_insert(0.0) += 1.0;
            let cst = (cls, t.style);
            *self.lexicon_cls_style.entry(cst).or_default().entry(t.lex).or_insert(0.0) += 1.0;
            *self.lexicon_cls_style_totals.entry(cst).or_insert(0.0) += 1.0;
            *self.lexicon_class.entry(cls).or_default().entry(t.lex).or_insert(0.0) += 1.0;
            *self.lexicon_class_totals.entry(cls).or_insert(0.0) += 1.0;

            if i > 0 {
                let prev = &tokens[i-1];

                // T2 — head imediato
                let head_idx = (i as i16 + t.syn_off) as usize;
                if head_idx < tokens.len() && head_idx != i {
                    let head = &tokens[head_idx];
                    let t2key = (head.morph, head.style, cls);
                    *self.agreement.entry(t2key).or_default()
                        .entry((t.morph, t.style)).or_insert(0.0) += 1.0;
                    *self.agreement_totals.entry(t2key).or_insert(0.0) += 1.0;

                    // T2 multi-hop: avô
                    let gp_idx = (head_idx as i16 + head.syn_off) as usize;
                    if gp_idx < tokens.len() && gp_idx != head_idx && gp_idx != i {
                        let gp = &tokens[gp_idx];
                        let gpkey = (gp.morph, gp.style, cls);
                        *self.agreement_gparent.entry(gpkey).or_default()
                            .entry((t.morph, t.style)).or_insert(0.0) += 1.0;
                        *self.agreement_gparent_totals.entry(gpkey).or_insert(0.0) += 1.0;
                    }
                }

                // T2 fallback sequencial
                let t2key_seq = (prev.morph, prev.style, cls);
                *self.agreement.entry(t2key_seq).or_default()
                    .entry((t.morph, t.style)).or_insert(0.0) += 1.0;
                *self.agreement_totals.entry(t2key_seq).or_insert(0.0) += 1.0;

                // T2 class-based fallback
                *self.agreement_class.entry(cls).or_default()
                    .entry((t.morph, t.style)).or_insert(0.0) += 1.0;
                *self.agreement_class_totals.entry(cls).or_insert(0.0) += 1.0;
            }
        }

        // Treinar hipercubo hierárquico
        self.hier.train(tokens);
    }

    // ─── Refinamento T2/T3 sobre um lex predito ───

    fn refine_token(&mut self, history: &[Token7], lex: u32) -> Token7 {
        if lex == 0 { return Token7::new(0, 0); }
        let temp = self.temperature.max(0.01);
        let explore = self.exploration_rate;

        let syn_history = if history.len() >= 2 { assign_dependencies(history) } else { history.to_vec() };
        let clause_depths = self.current_clause_depths(&syn_history);

        let cls = self.lex_to_class.get(&lex).copied().unwrap_or_else(|| {
            self.guess_class(&syn_history, temp, explore)
        });

        let mut extended = syn_history.clone();
        extended.push(Token7::new(lex, cls as u16));
        let dep = assign_dependencies(&extended);
        let temp_tok = dep.last().unwrap();
        let syn_off = temp_tok.syn_off;
        let syn_func = temp_tok.syn_func;

        let (morph_detail, style) = self.sample_features_multi_hop(&syn_history, syn_off, cls, &clause_depths);
        let morph = (cls as u16) | morph_detail;
        let lex = self.sample_lex_full(morph, syn_func, style, cls, lex);

        Token7::new(lex, morph).with_syn(syn_off, syn_func).with_style(style)
    }

    // ─── Sample token (CUBO clause para retrocompatibilidade) ───

    pub fn sample_token(&mut self, history: &[Token7]) -> Token7 {
        let syn_history = if history.len() >= 2 { assign_dependencies(history) } else { history.to_vec() };
        let clause_depths = self.current_clause_depths(&syn_history);
        let temp = self.temperature.max(0.01);
        let explore = self.exploration_rate;

        let lex = if self.graph_alpha > 0.0 && !self.graph.is_empty() {
            self.hier.clause.sample_lex_modulated(&syn_history, &clause_depths, temp, explore, &mut self.rng, Some(&self.graph), self.graph_alpha)
        } else {
            self.hier.clause.sample_lex(&syn_history, &clause_depths, temp, explore, &mut self.rng)
        };
        if lex == 0 { return Token7::new(0, 0); }
        self.refine_token(&syn_history, lex)
    }

    fn current_clause_depths(&self, history: &[Token7]) -> Vec<u8> {
        // Recompute depths for full history efficiently
        // For generation, we track incrementally
        if history.is_empty() { return vec![]; }
        // Just use compute for now (small history)
        Self::compute_clause_depths(history)
    }

    fn guess_class(&mut self, history: &[Token7], temp: f32, explore: f32) -> u8 {
        // Usar hipercubo para inferir classe do contexto
        let clause_depths = self.current_clause_depths(history);
        let lex = self.hier.clause.sample_lex(history, &clause_depths, temp, explore, &mut self.rng);
        if let Some(&cls) = self.lex_to_class.get(&lex) {
            return cls;
        }
        // Fallback: último POS do histórico
        history.last().map(|t| t.morph_class()).unwrap_or(0)
    }

    // ─── T2 multi-hop: head + avô ───

    fn sample_features_multi_hop(&mut self, history: &[Token7], syn_off: i16, curr_cls: u8, _depths: &[u8]) -> (u16, u16) {
        let temp = self.temperature.max(0.01);
        let explore = self.exploration_rate;

        let head_idx = history.len() as i16 + syn_off;
        let head = if syn_off < 0 && head_idx >= 0 && (head_idx as usize) < history.len() {
            Some(&history[head_idx as usize])
        } else {
            history.last()
        };

        let (head_morph, head_style) = head.map(|h| (h.morph, h.style)).unwrap_or((curr_cls as u16, 0));

        // Tentar propagação do avô primeiro (mais distante → mais contexto)
        if let Some(h) = head {
            let gp_idx = (head_idx as usize) as i16 + h.syn_off;
            if gp_idx >= 0 && (gp_idx as usize) < history.len() && (gp_idx as usize) != head_idx as usize {
                let gp = &history[gp_idx as usize];
                let gpkey = (gp.morph, gp.style, curr_cls);
                if let Some(res) = self.sample_t2_key(&gpkey, temp, explore) {
                    // Interpolar com head: 0.4 gp + 0.6 head (soft)
                    let head_res = self.sample_t2_key(&(head_morph, head_style, curr_cls), temp, explore);
                    if let Some((hm, hs)) = head_res {
                        let md = ((res.0 as f32 * 0.4 + hm as f32 * 0.6) as u16) & !0x07;
                        let st = (res.1 as f32 * 0.4 + hs as f32 * 0.6) as u16;
                        return (md, st);
                    }
                    return res;
                }
            }
        }

        // Head imediato
        let key = (head_morph, head_style, curr_cls);
        if let Some(res) = self.sample_t2_key(&key, temp, explore) { return res; }

        // Fallback classe
        if let Some(res) = self.sample_t2_class(curr_cls, temp, explore) { return res; }

        // Copiar detalhe do head
        ((head_morph & !0x07), head_style)
    }

    fn sample_t2_key(&mut self, key: &(u16, u16, u8), temp: f32, explore: f32) -> Option<(u16, u16)> {
        let candidates = self.agreement.get(key)?;
        let total = *self.agreement_totals.get(key)?;
        Self::sample_t2_dist(&mut self.rng, candidates, total, temp, explore)
    }

    fn sample_t2_class(&mut self, cls: u8, temp: f32, explore: f32) -> Option<(u16, u16)> {
        let candidates = self.agreement_class.get(&cls)?;
        let total = *self.agreement_class_totals.get(&cls)?;
        Self::sample_t2_dist(&mut self.rng, candidates, total, temp, explore)
    }

    fn sample_t2_dist(rng: &mut fastrand::Rng, candidates: &HashMap<(u16, u16), f32>, total: f32, temp: f32, explore: f32) -> Option<(u16, u16)> {
        if total < 1.0 { return None; }
        if rng.f32() < explore {
            let filtered: Vec<(u16, u16)> = candidates.keys().copied().collect();
            return Some(if filtered.is_empty() { return None; } else { filtered[rng.usize(0..filtered.len())] });
        }
        let mut sum = 0.0f64;
        let mut probs: Vec<((u16, u16), f64)> = Vec::new();
        for (&v, &count) in candidates {
            let w = (count as f64).powf(1.0 / temp as f64);
            if w > 0.0 { sum += w; probs.push((v, w)); }
        }
        if sum <= 0.0 { return None; }
        let mut r = rng.f64() * sum;
        for &(v, w) in &probs { r -= w; if r <= 0.0 { return Some(v); } }
        probs.last().map(|x| x.0)
    }

    // ─── T3 cascata (inalterada) ───

    pub fn sample_lex_full(&mut self, morph: u16, syn_func: u8, style: u16, cls: u8, candidate: u32) -> u32 {
        let temp = self.temperature.max(0.01);
        let explore = self.exploration_rate;

        // Se o candidato do CUBO existe na T3 exata, usá-lo
        let key = (morph, syn_func, style);
        if let Some(map) = self.lexicon.get(&key) {
            if map.contains_key(&candidate) { return candidate; }
            if let Some(id) = Self::sample_lex_map(&mut self.rng, map, *self.lexicon_totals.get(&key).unwrap_or(&0.0), temp, explore) {
                return id;
            }
        }
        // Próximos níveis da cascata — preferir candidato se existir
        let css = (cls, syn_func, style);
        if let Some(map) = self.lexicon_cls_syn_style.get(&css) {
            if map.contains_key(&candidate) { return candidate; }
            if let Some(id) = Self::sample_lex_map(&mut self.rng, map, *self.lexicon_cls_syn_style_totals.get(&css).unwrap_or(&0.0), temp, explore) { return id; }
        }
        let cs = (cls, syn_func);
        if let Some(map) = self.lexicon_cls_syn.get(&cs) {
            if map.contains_key(&candidate) { return candidate; }
            if let Some(id) = Self::sample_lex_map(&mut self.rng, map, *self.lexicon_cls_syn_totals.get(&cs).unwrap_or(&0.0), temp, explore) { return id; }
        }
        let cst = (cls, style);
        if let Some(map) = self.lexicon_cls_style.get(&cst) {
            if map.contains_key(&candidate) { return candidate; }
            if let Some(id) = Self::sample_lex_map(&mut self.rng, map, *self.lexicon_cls_style_totals.get(&cst).unwrap_or(&0.0), temp, explore) { return id; }
        }
        if let Some(map) = self.lexicon_class.get(&cls) {
            if map.contains_key(&candidate) { return candidate; }
            if let Some(id) = Self::sample_lex_map(&mut self.rng, map, *self.lexicon_class_totals.get(&cls).unwrap_or(&0.0), temp, explore) { return id; }
        }
        // Último recurso: devolver o candidato do CUBO
        candidate
    }

    fn sample_lex_map(rng: &mut fastrand::Rng, candidates: &HashMap<u32, f32>, total: f32, temp: f32, explore: f32) -> Option<u32> {
        if total < 1.0 { return None; }
        if rng.f32() < explore {
            let filtered: Vec<u32> = candidates.keys().copied().collect();
            if filtered.is_empty() { return None; }
            return Some(filtered[rng.usize(0..filtered.len())]);
        }
        let mut sum = 0.0f64;
        let mut probs: Vec<(u32, f64)> = Vec::new();
        for (&id, &count) in candidates {
            let w = (count as f64).powf(1.0 / temp as f64);
            if w > 0.0 { sum += w; probs.push((id, w)); }
        }
        if sum <= 0.0 { return None; }
        let mut r = rng.f64() * sum;
        for &(id, w) in &probs { r -= w; if r <= 0.0 { return Some(id); } }
        probs.last().map(|x| x.0)
    }

    // ─── Geração (hierárquica top-down) ───

    pub fn generate(&mut self, seed: &[Token7], max_len: usize) -> Vec<Token7> {
        let raw = self.hier.generate(seed, max_len);
        let mut out = seed.to_vec();
        for tok in raw.iter().skip(seed.len()) {
            let refined = self.refine_token(&out, tok.lex);
            out.push(refined);
            if refined.lex == 0 { break; }
        }
        out
    }

    /// Geração usando SNN para seleção lexical ao invés da cascata T3
    pub fn generate_with_snn(&mut self, seed: &[Token7], max_len: usize) -> Vec<Token7> {
        if self.snn.is_none() {
            return self.generate(seed, max_len);
        }

        let mut out = seed.to_vec();

        for _ in 0..max_len {
            // 1. CUBO prediz lex_id (modulação GRAPH)
            let syn_history = if out.len() >= 2 { assign_dependencies(&out) } else { out.clone() };
            let clause_depths = Self::compute_clause_depths(&syn_history);
            let temp = self.temperature.max(0.01);
            let explore = self.exploration_rate;

            let cubo_lex = if self.graph_alpha > 0.0 && !self.graph.is_empty() {
                self.hier.clause.sample_lex_modulated(&syn_history, &clause_depths, temp, explore, &mut self.rng, Some(&self.graph), self.graph_alpha)
            } else {
                self.hier.clause.sample_lex(&syn_history, &clause_depths, temp, explore, &mut self.rng)
            };

            if cubo_lex == 0 { break; }

            // 2. refine_token já faz T2 + T3: usa o lex do CUBO e refina morph/style
            let refined = self.refine_token(&out, cubo_lex);

            // 3. SNN substitui o T3: recomputa lex com SNN
            let snn_lex = self.snn_sample(refined.morph, refined.syn_func, refined.style, cubo_lex);
            let final_lex = if snn_lex != 0 { snn_lex } else { cubo_lex };

            let final_token = Token7::new(final_lex, refined.morph)
                .with_syn(refined.syn_off, refined.syn_func)
                .with_style(refined.style);
            out.push(final_token);
        }

        out
    }

    // ─── Reforço ───

    pub fn reinforce(&mut self, tokens: &[Token7], delta: f32) {
        let depths = Self::compute_clause_depths(tokens);
        self.hier.clause.reinforce(tokens, &depths, delta);
        for i in 1..tokens.len() {
            let t = &tokens[i];
            let prev = &tokens[i-1];

            let t3key = (t.morph, t.syn_func, t.style);
            if let Some(map) = self.lexicon.get_mut(&t3key) {
                if let Some(c) = map.get_mut(&t.lex) { *c = (*c + delta).max(0.01); }
            }
            let cls = t.morph_class();
            let css = (cls, t.syn_func, t.style);
            if let Some(map) = self.lexicon_cls_syn_style.get_mut(&css) {
                if let Some(c) = map.get_mut(&t.lex) { *c = (*c + delta).max(0.01); }
            }
            let cs = (cls, t.syn_func);
            if let Some(map) = self.lexicon_cls_syn.get_mut(&cs) {
                if let Some(c) = map.get_mut(&t.lex) { *c = (*c + delta).max(0.01); }
            }
            let cst = (cls, t.style);
            if let Some(map) = self.lexicon_cls_style.get_mut(&cst) {
                if let Some(c) = map.get_mut(&t.lex) { *c = (*c + delta).max(0.01); }
            }
            if let Some(map) = self.lexicon_class.get_mut(&cls) {
                if let Some(c) = map.get_mut(&t.lex) { *c = (*c + delta).max(0.01); }
            }

            let head_idx = (i as i16 + t.syn_off) as usize;
            if head_idx < tokens.len() && head_idx != i {
                let head = &tokens[head_idx];
                let t2key = (head.morph, head.style, cls);
                if let Some(map) = self.agreement.get_mut(&t2key) {
                    if let Some(c) = map.get_mut(&(t.morph, t.style)) { *c = (*c + delta).max(0.01); }
                }
            }
            let t2key_seq = (prev.morph, prev.style, cls);
            if let Some(map) = self.agreement.get_mut(&t2key_seq) {
                if let Some(c) = map.get_mut(&(t.morph, t.style)) { *c = (*c + delta).max(0.01); }
            }
        }
    }

    // ─── Save / Load ───

    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let mut data = Vec::new();
        data.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        data.extend_from_slice(&(self.syntax_order as u32).to_le_bytes());
        data.extend_from_slice(&self.temperature.to_le_bytes());
        data.extend_from_slice(&self.exploration_rate.to_le_bytes());

        // Hipercubo hierárquico
        self.hier.save(&mut data);

        // T2
        Self::save_t2(&mut data, &self.agreement, &self.agreement_totals);
        Self::save_t2_class(&mut data, &self.agreement_class, &self.agreement_class_totals);
        Self::save_t2(&mut data, &self.agreement_gparent, &self.agreement_gparent_totals);

        // T3
        Self::save_t3(&mut data, &self.lexicon, &self.lexicon_totals);
        Self::save_t3_css(&mut data, &self.lexicon_cls_syn_style, &self.lexicon_cls_syn_style_totals);
        Self::save_t3_cs(&mut data, &self.lexicon_cls_syn, &self.lexicon_cls_syn_totals);
        Self::save_t3_cst(&mut data, &self.lexicon_cls_style, &self.lexicon_cls_style_totals);
        Self::save_lex_class(&mut data, &self.lexicon_class, &self.lexicon_class_totals);

        // lex_to_class
        Self::save_lex_map_u32_u8(&mut data, &self.lex_to_class);

        // graph
        Self::save_lex_map_u32_u32(&mut data, &self.graph);
        data.extend_from_slice(&self.graph_alpha.to_le_bytes());

        fs::write(path, &data)
    }

    pub fn load(&mut self, path: &str) -> std::io::Result<()> {
        let data = fs::read(path)?;
        let mut off = 0;

        let version = u32::from_le_bytes(data[off..off+4].try_into().unwrap());
        off += 4;
        if version != FORMAT_VERSION {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData,
                format!("formato {} incompatível (esperado {})", version, FORMAT_VERSION)));
        }

        self.syntax_order = u32::from_le_bytes(data[off..off+4].try_into().unwrap()) as usize;
        off += 4;
        self.temperature = f32::from_le_bytes(data[off..off+4].try_into().unwrap());
        off += 4;
        self.exploration_rate = f32::from_le_bytes(data[off..off+4].try_into().unwrap());
        off += 4;

        self.hier.load(&data, &mut off);

        Self::load_t2(&data, &mut off, &mut self.agreement, &mut self.agreement_totals);
        Self::load_t2_class(&data, &mut off, &mut self.agreement_class, &mut self.agreement_class_totals);
        Self::load_t2(&data, &mut off, &mut self.agreement_gparent, &mut self.agreement_gparent_totals);

        Self::load_t3(&data, &mut off, &mut self.lexicon, &mut self.lexicon_totals);
        Self::load_t3_css(&data, &mut off, &mut self.lexicon_cls_syn_style, &mut self.lexicon_cls_syn_style_totals);
        Self::load_t3_cs(&data, &mut off, &mut self.lexicon_cls_syn, &mut self.lexicon_cls_syn_totals);
        Self::load_t3_cst(&data, &mut off, &mut self.lexicon_cls_style, &mut self.lexicon_cls_style_totals);
        Self::load_lex_class(&data, &mut off, &mut self.lexicon_class, &mut self.lexicon_class_totals);

        // lex_to_class
        Self::load_lex_map_u32_u8(&data, &mut off, &mut self.lex_to_class);

        // graph
        Self::load_lex_map_u32_u32(&data, &mut off, &mut self.graph);
        if off + 4 <= data.len() {
            self.graph_alpha = f32::from_le_bytes(data[off..off+4].try_into().unwrap_or([0;4]));
            off += 4;
        }

        Ok(())
    }

    // ─── Serializers ───

    fn save_t2(data: &mut Vec<u8>, map: &HashMap<(u16, u16, u8), HashMap<(u16, u16), f32>>, totals: &HashMap<(u16, u16, u8), f32>) {
        data.extend_from_slice(&(map.len() as u32).to_le_bytes());
        for (&(m1, s1, c), nexts) in map {
            data.extend_from_slice(&m1.to_le_bytes());
            data.extend_from_slice(&s1.to_le_bytes());
            data.push(c);
            data.extend_from_slice(&totals.get(&(m1, s1, c)).copied().unwrap_or(0.0).to_le_bytes());
            let n = nexts.len() as u32;
            data.extend_from_slice(&n.to_le_bytes());
            for (&(m2, s2), &v) in nexts {
                data.extend_from_slice(&m2.to_le_bytes());
                data.extend_from_slice(&s2.to_le_bytes());
                data.extend_from_slice(&v.to_le_bytes());
            }
        }
    }

    fn load_t2(data: &[u8], off: &mut usize, map: &mut HashMap<(u16, u16, u8), HashMap<(u16, u16), f32>>, totals: &mut HashMap<(u16, u16, u8), f32>) {
        let n = u32::from_le_bytes(data[*off..*off+4].try_into().unwrap()) as usize;
        *off += 4;
        for _ in 0..n {
            let m1 = u16::from_le_bytes(data[*off..*off+2].try_into().unwrap()); *off += 2;
            let s1 = u16::from_le_bytes(data[*off..*off+2].try_into().unwrap()); *off += 2;
            let c = data[*off]; *off += 1;
            let total = f32::from_le_bytes(data[*off..*off+4].try_into().unwrap()); *off += 4;
            let n_next = u32::from_le_bytes(data[*off..*off+4].try_into().unwrap()) as usize;
            *off += 4;
            let mut nexts = HashMap::with_capacity(n_next);
            for _ in 0..n_next {
                let m2 = u16::from_le_bytes(data[*off..*off+2].try_into().unwrap()); *off += 2;
                let s2 = u16::from_le_bytes(data[*off..*off+2].try_into().unwrap()); *off += 2;
                let v = f32::from_le_bytes(data[*off..*off+4].try_into().unwrap()); *off += 4;
                nexts.insert((m2, s2), v);
            }
            totals.insert((m1, s1, c), total);
            map.insert((m1, s1, c), nexts);
        }
    }

    fn save_t2_class(data: &mut Vec<u8>, map: &HashMap<u8, HashMap<(u16, u16), f32>>, totals: &HashMap<u8, f32>) {
        data.extend_from_slice(&(map.len() as u32).to_le_bytes());
        for (&c, nexts) in map {
            data.push(c);
            data.extend_from_slice(&totals.get(&c).copied().unwrap_or(0.0).to_le_bytes());
            let n = nexts.len() as u32;
            data.extend_from_slice(&n.to_le_bytes());
            for (&(m2, s2), &v) in nexts {
                data.extend_from_slice(&m2.to_le_bytes());
                data.extend_from_slice(&s2.to_le_bytes());
                data.extend_from_slice(&v.to_le_bytes());
            }
        }
    }

    fn load_t2_class(data: &[u8], off: &mut usize, map: &mut HashMap<u8, HashMap<(u16, u16), f32>>, totals: &mut HashMap<u8, f32>) {
        let n = u32::from_le_bytes(data[*off..*off+4].try_into().unwrap()) as usize;
        *off += 4;
        for _ in 0..n {
            let c = data[*off]; *off += 1;
            let total = f32::from_le_bytes(data[*off..*off+4].try_into().unwrap()); *off += 4;
            let n_next = u32::from_le_bytes(data[*off..*off+4].try_into().unwrap()) as usize;
            *off += 4;
            let mut nexts = HashMap::with_capacity(n_next);
            for _ in 0..n_next {
                let m2 = u16::from_le_bytes(data[*off..*off+2].try_into().unwrap()); *off += 2;
                let s2 = u16::from_le_bytes(data[*off..*off+2].try_into().unwrap()); *off += 2;
                let v = f32::from_le_bytes(data[*off..*off+4].try_into().unwrap()); *off += 4;
                nexts.insert((m2, s2), v);
            }
            totals.insert(c, total);
            map.insert(c, nexts);
        }
    }

    fn save_t3(data: &mut Vec<u8>, map: &HashMap<(u16, u8, u16), HashMap<u32, f32>>, totals: &HashMap<(u16, u8, u16), f32>) {
        data.extend_from_slice(&(map.len() as u32).to_le_bytes());
        for (&(m, sf, st), nexts) in map {
            data.extend_from_slice(&m.to_le_bytes());
            data.push(sf);
            data.extend_from_slice(&st.to_le_bytes());
            data.extend_from_slice(&totals.get(&(m, sf, st)).copied().unwrap_or(0.0).to_le_bytes());
            let n = nexts.len() as u32;
            data.extend_from_slice(&n.to_le_bytes());
            for (&id, &v) in nexts {
                data.extend_from_slice(&id.to_le_bytes());
                data.extend_from_slice(&v.to_le_bytes());
            }
        }
    }

    fn load_t3(data: &[u8], off: &mut usize, map: &mut HashMap<(u16, u8, u16), HashMap<u32, f32>>, totals: &mut HashMap<(u16, u8, u16), f32>) {
        let n = u32::from_le_bytes(data[*off..*off+4].try_into().unwrap()) as usize;
        *off += 4;
        for _ in 0..n {
            let m = u16::from_le_bytes(data[*off..*off+2].try_into().unwrap()); *off += 2;
            let sf = data[*off]; *off += 1;
            let st = u16::from_le_bytes(data[*off..*off+2].try_into().unwrap()); *off += 2;
            let total = f32::from_le_bytes(data[*off..*off+4].try_into().unwrap()); *off += 4;
            let n_next = u32::from_le_bytes(data[*off..*off+4].try_into().unwrap()) as usize;
            *off += 4;
            let mut nexts = HashMap::with_capacity(n_next);
            for _ in 0..n_next {
                let id = u32::from_le_bytes(data[*off..*off+4].try_into().unwrap()); *off += 4;
                let v = f32::from_le_bytes(data[*off..*off+4].try_into().unwrap()); *off += 4;
                nexts.insert(id, v);
            }
            totals.insert((m, sf, st), total);
            map.insert((m, sf, st), nexts);
        }
    }

    fn save_t3_css(data: &mut Vec<u8>, map: &HashMap<(u8, u8, u16), HashMap<u32, f32>>, totals: &HashMap<(u8, u8, u16), f32>) {
        data.extend_from_slice(&(map.len() as u32).to_le_bytes());
        for (&(c, sf, st), nexts) in map {
            data.push(c); data.push(sf);
            data.extend_from_slice(&st.to_le_bytes());
            data.extend_from_slice(&totals.get(&(c, sf, st)).copied().unwrap_or(0.0).to_le_bytes());
            let n = nexts.len() as u32;
            data.extend_from_slice(&n.to_le_bytes());
            for (&id, &v) in nexts {
                data.extend_from_slice(&id.to_le_bytes());
                data.extend_from_slice(&v.to_le_bytes());
            }
        }
    }

    fn load_t3_css(data: &[u8], off: &mut usize, map: &mut HashMap<(u8, u8, u16), HashMap<u32, f32>>, totals: &mut HashMap<(u8, u8, u16), f32>) {
        let n = u32::from_le_bytes(data[*off..*off+4].try_into().unwrap()) as usize;
        *off += 4;
        for _ in 0..n {
            let c = data[*off]; *off += 1;
            let sf = data[*off]; *off += 1;
            let st = u16::from_le_bytes(data[*off..*off+2].try_into().unwrap()); *off += 2;
            let total = f32::from_le_bytes(data[*off..*off+4].try_into().unwrap()); *off += 4;
            let n_next = u32::from_le_bytes(data[*off..*off+4].try_into().unwrap()) as usize;
            *off += 4;
            let mut nexts = HashMap::with_capacity(n_next);
            for _ in 0..n_next {
                let id = u32::from_le_bytes(data[*off..*off+4].try_into().unwrap()); *off += 4;
                let v = f32::from_le_bytes(data[*off..*off+4].try_into().unwrap()); *off += 4;
                nexts.insert(id, v);
            }
            totals.insert((c, sf, st), total);
            map.insert((c, sf, st), nexts);
        }
    }

    fn save_t3_cs(data: &mut Vec<u8>, map: &HashMap<(u8, u8), HashMap<u32, f32>>, totals: &HashMap<(u8, u8), f32>) {
        data.extend_from_slice(&(map.len() as u32).to_le_bytes());
        for (&(c, sf), nexts) in map {
            data.push(c); data.push(sf);
            data.extend_from_slice(&totals.get(&(c, sf)).copied().unwrap_or(0.0).to_le_bytes());
            let n = nexts.len() as u32;
            data.extend_from_slice(&n.to_le_bytes());
            for (&id, &v) in nexts {
                data.extend_from_slice(&id.to_le_bytes());
                data.extend_from_slice(&v.to_le_bytes());
            }
        }
    }

    fn load_t3_cs(data: &[u8], off: &mut usize, map: &mut HashMap<(u8, u8), HashMap<u32, f32>>, totals: &mut HashMap<(u8, u8), f32>) {
        let n = u32::from_le_bytes(data[*off..*off+4].try_into().unwrap()) as usize;
        *off += 4;
        for _ in 0..n {
            let c = data[*off]; *off += 1;
            let sf = data[*off]; *off += 1;
            let total = f32::from_le_bytes(data[*off..*off+4].try_into().unwrap()); *off += 4;
            let n_next = u32::from_le_bytes(data[*off..*off+4].try_into().unwrap()) as usize;
            *off += 4;
            let mut nexts = HashMap::with_capacity(n_next);
            for _ in 0..n_next {
                let id = u32::from_le_bytes(data[*off..*off+4].try_into().unwrap()); *off += 4;
                let v = f32::from_le_bytes(data[*off..*off+4].try_into().unwrap()); *off += 4;
                nexts.insert(id, v);
            }
            totals.insert((c, sf), total);
            map.insert((c, sf), nexts);
        }
    }

    fn save_t3_cst(data: &mut Vec<u8>, map: &HashMap<(u8, u16), HashMap<u32, f32>>, totals: &HashMap<(u8, u16), f32>) {
        data.extend_from_slice(&(map.len() as u32).to_le_bytes());
        for (&(c, st), nexts) in map {
            data.push(c);
            data.extend_from_slice(&st.to_le_bytes());
            data.extend_from_slice(&totals.get(&(c, st)).copied().unwrap_or(0.0).to_le_bytes());
            let n = nexts.len() as u32;
            data.extend_from_slice(&n.to_le_bytes());
            for (&id, &v) in nexts {
                data.extend_from_slice(&id.to_le_bytes());
                data.extend_from_slice(&v.to_le_bytes());
            }
        }
    }

    fn load_t3_cst(data: &[u8], off: &mut usize, map: &mut HashMap<(u8, u16), HashMap<u32, f32>>, totals: &mut HashMap<(u8, u16), f32>) {
        let n = u32::from_le_bytes(data[*off..*off+4].try_into().unwrap()) as usize;
        *off += 4;
        for _ in 0..n {
            let c = data[*off]; *off += 1;
            let st = u16::from_le_bytes(data[*off..*off+2].try_into().unwrap()); *off += 2;
            let total = f32::from_le_bytes(data[*off..*off+4].try_into().unwrap()); *off += 4;
            let n_next = u32::from_le_bytes(data[*off..*off+4].try_into().unwrap()) as usize;
            *off += 4;
            let mut nexts = HashMap::with_capacity(n_next);
            for _ in 0..n_next {
                let id = u32::from_le_bytes(data[*off..*off+4].try_into().unwrap()); *off += 4;
                let v = f32::from_le_bytes(data[*off..*off+4].try_into().unwrap()); *off += 4;
                nexts.insert(id, v);
            }
            totals.insert((c, st), total);
            map.insert((c, st), nexts);
        }
    }

    fn save_lex_map_u32_u8(data: &mut Vec<u8>, map: &HashMap<u32, u8>) {
        data.extend_from_slice(&(map.len() as u32).to_le_bytes());
        for (&k, &v) in map {
            data.extend_from_slice(&k.to_le_bytes());
            data.push(v);
        }
    }

    fn load_lex_map_u32_u8(data: &[u8], off: &mut usize, map: &mut HashMap<u32, u8>) {
        let n = u32::from_le_bytes(data[*off..*off+4].try_into().unwrap()) as usize;
        *off += 4;
        for _ in 0..n {
            let k = u32::from_le_bytes(data[*off..*off+4].try_into().unwrap()); *off += 4;
            let v = data[*off]; *off += 1;
            map.insert(k, v);
        }
    }

    fn save_lex_map_u32_u32(data: &mut Vec<u8>, map: &HashMap<u32, u32>) {
        data.extend_from_slice(&(map.len() as u32).to_le_bytes());
        for (&k, &v) in map {
            data.extend_from_slice(&k.to_le_bytes());
            data.extend_from_slice(&v.to_le_bytes());
        }
    }

    fn load_lex_map_u32_u32(data: &[u8], off: &mut usize, map: &mut HashMap<u32, u32>) {
        let n = u32::from_le_bytes(data[*off..*off+4].try_into().unwrap()) as usize;
        *off += 4;
        for _ in 0..n {
            let k = u32::from_le_bytes(data[*off..*off+4].try_into().unwrap()); *off += 4;
            let v = u32::from_le_bytes(data[*off..*off+4].try_into().unwrap()); *off += 4;
            map.insert(k, v);
        }
    }

    fn save_lex_class(data: &mut Vec<u8>, map: &HashMap<u8, HashMap<u32, f32>>, totals: &HashMap<u8, f32>) {
        data.extend_from_slice(&(map.len() as u32).to_le_bytes());
        for (&c, nexts) in map {
            data.push(c);
            data.extend_from_slice(&totals.get(&c).copied().unwrap_or(0.0).to_le_bytes());
            let n = nexts.len() as u32;
            data.extend_from_slice(&n.to_le_bytes());
            for (&id, &v) in nexts {
                data.extend_from_slice(&id.to_le_bytes());
                data.extend_from_slice(&v.to_le_bytes());
            }
        }
    }

    fn load_lex_class(data: &[u8], off: &mut usize, map: &mut HashMap<u8, HashMap<u32, f32>>, totals: &mut HashMap<u8, f32>) {
        let n = u32::from_le_bytes(data[*off..*off+4].try_into().unwrap()) as usize;
        *off += 4;
        for _ in 0..n {
            let c = data[*off]; *off += 1;
            let total = f32::from_le_bytes(data[*off..*off+4].try_into().unwrap()); *off += 4;
            let n_next = u32::from_le_bytes(data[*off..*off+4].try_into().unwrap()) as usize;
            *off += 4;
            let mut nexts = HashMap::with_capacity(n_next);
            for _ in 0..n_next {
                let id = u32::from_le_bytes(data[*off..*off+4].try_into().unwrap()); *off += 4;
                let v = f32::from_le_bytes(data[*off..*off+4].try_into().unwrap()); *off += 4;
                nexts.insert(id, v);
            }
            totals.insert(c, total);
            map.insert(c, nexts);
        }
    }

    // ═══════════════════════════════════════════════════════
    // GGUF Export / Import
    // ═══════════════════════════════════════════════════════

    /// Export grammar to GGUF format. Tensors use F32 for portability.
    pub fn save_gguf(&self, path: &str, compiler_lexicon: &HashMap<String, (u32, u16, u8, u16)>, compiler_roots: &[String]) -> std::io::Result<()> {
        use crate::quant::gguf::{GgufWriter, GgufKv, GgufValue, GgufTensorInfo};
        use crate::quant::QuantType;

        // ─── 1. Build all tensor data sections first (no padding) ───
        let mut sections: Vec<(&str, Vec<u8>)> = Vec::new();

        // Lexicon
        let mut lex_data = Vec::new();
        for (_form, &(id, morph, cls, sty)) in compiler_lexicon {
            lex_data.extend_from_slice(&id.to_le_bytes());
            lex_data.extend_from_slice(&morph.to_le_bytes());
            lex_data.extend_from_slice(&(cls as u32).to_le_bytes());
            lex_data.extend_from_slice(&(sty as u32).to_le_bytes());
            lex_data.extend_from_slice(&0u32.to_le_bytes());
            lex_data.extend_from_slice(&0u32.to_le_bytes());
            lex_data.extend_from_slice(&0u32.to_le_bytes());
            lex_data.extend_from_slice(&0u32.to_le_bytes());
        }
        sections.push(("lexicon.forms", lex_data));

        // Roots
        let mut roots_data = Vec::new();
        for root in compiler_roots {
            let id = compiler_lexicon.get(root).map(|v| v.0).unwrap_or(0);
            roots_data.extend_from_slice(&id.to_le_bytes());
        }
        sections.push(("lexicon.roots", roots_data));

        // GRAPH
        let mut graph_data = Vec::new();
        for (&lex_id, &sig) in &self.graph {
            graph_data.extend_from_slice(&lex_id.to_le_bytes());
            graph_data.extend_from_slice(&sig.to_le_bytes());
        }
        sections.push(("graph.data", graph_data));

        // lex_to_class
        let mut ltc_data = Vec::new();
        for (&lex_id, &cls) in &self.lex_to_class {
            ltc_data.extend_from_slice(&lex_id.to_le_bytes());
            ltc_data.extend_from_slice(&(cls as u32).to_le_bytes());
        }
        sections.push(("lex_to_class.data", ltc_data));

        // T2 head
        sections.push(("t2.head.data", Self::serialize_t2_flat(&self.agreement, &self.agreement_totals)));
        // T2 gparent
        sections.push(("t2.gparent.data", Self::serialize_t2_flat(&self.agreement_gparent, &self.agreement_gparent_totals)));
        // T2 class
        sections.push(("t2.class.data", Self::serialize_t2_class_flat(&self.agreement_class, &self.agreement_class_totals)));

        // T3 exact
        sections.push(("t3.exact.data", Self::serialize_t3_flat(&self.lexicon, &self.lexicon_totals)));
        // T3 CSS
        sections.push(("t3.css.data", Self::serialize_t3_css_flat(&self.lexicon_cls_syn_style, &self.lexicon_cls_syn_style_totals)));
        // T3 CS
        sections.push(("t3.cs.data", Self::serialize_t3_cs_flat(&self.lexicon_cls_syn, &self.lexicon_cls_syn_totals)));
        // T3 CST
        sections.push(("t3.cst.data", Self::serialize_t3_cst_flat(&self.lexicon_cls_style, &self.lexicon_cls_style_totals)));
        // T3 class
        sections.push(("t3.class.data", Self::serialize_t3_class_flat(&self.lexicon_class, &self.lexicon_class_totals)));

        // CUBO blobs
        sections.push(("cubo.clause.blob", Self::serialize_cubo_blob(&self.hier.clause)));
        sections.push(("cubo.sentence.blob", Self::serialize_cubo_blob(&self.hier.sentence)));
        sections.push(("cubo.paragraph.blob", Self::serialize_cubo_blob(&self.hier.paragraph)));
        sections.push(("cubo.text.blob", Self::serialize_cubo_blob(&self.hier.text)));

        // SNN weights (se disponível)
        let has_snn = self.snn.is_some();
        if let Some(ref snn) = self.snn {
            // snn.synapses_ih: flat [n_input * n_hidden]
            let ih_flat = snn.flat_ih();
            let n_input = snn.input_neurons.len();
            let n_hidden = snn.hidden_neurons.len();
            let mut ih_data = Vec::with_capacity(ih_flat.len() * 4);
            for v in &ih_flat { ih_data.extend_from_slice(&v.to_le_bytes()); }
            sections.push(("snn.synapses_ih", ih_data));

            // snn.synapses_ho: flat [n_hidden * n_output]
            let ho_flat = snn.flat_ho();
            let n_out = snn.output_neurons.len();
            let mut ho_data = Vec::with_capacity(ho_flat.len() * 4);
            for v in &ho_flat { ho_data.extend_from_slice(&v.to_le_bytes()); }
            sections.push(("snn.synapses_ho", ho_data));

            // snn.output_labels: [n_output]
            let labels_flat = snn.flat_labels();
            let mut labels_data = Vec::with_capacity(labels_flat.len() * 4);
            for v in &labels_flat { labels_data.extend_from_slice(&v.to_le_bytes()); }
            sections.push(("snn.output_labels", labels_data));
        }

        // ─── 2. Compute offsets and shapes ───
        let mut data_off: u64 = 0;
        let mut tensor_infos = Vec::new();
        let n_lex = compiler_lexicon.len() as u64;
        let n_roots = compiler_roots.iter().filter(|r| !r.is_empty()).count() as u64;
        let n_graph = self.graph.len() as u64;
        let n_ltc = self.lex_to_class.len() as u64;
        let n_tensors = sections.len() as u64;

        for (i, (name, data)) in sections.iter().enumerate() {
            let numel = data.len() / 4;
            let n_dims = match *name {
                "lexicon.forms" => vec![n_lex, 8],
                "lexicon.roots" => vec![n_roots],
                "graph.data" => vec![n_graph, 2],
                "lex_to_class.data" => vec![n_ltc, 2],
                "snn.synapses_ih" => {
                    if let Some(ref snn) = self.snn {
                        vec![snn.input_neurons.len() as u64, snn.hidden_neurons.len() as u64]
                    } else { vec![numel as u64] }
                }
                "snn.synapses_ho" => {
                    if let Some(ref snn) = self.snn {
                        vec![snn.hidden_neurons.len() as u64, snn.output_neurons.len() as u64]
                    } else { vec![numel as u64] }
                }
                "snn.output_labels" => {
                    if let Some(ref snn) = self.snn {
                        vec![snn.output_neurons.len() as u64]
                    } else { vec![numel as u64] }
                }
                _ => {
                    let count = numel as u64;
                    let cols: u64 = match *name {
                        "t2.head.data" | "t2.gparent.data" => 6,
                        "t2.class.data" | "t3.exact.data" | "t3.css.data" => 5,
                        "t3.cs.data" | "t3.cst.data" => 4,
                        "t3.class.data" => 3,
                        _ => 1,
                    };
                    if cols > 1 && count > 0 { vec![count / cols, cols] } else { vec![count] }
                }
            };
            tensor_infos.push(GgufTensorInfo {
                name: name.to_string(),
                n_dims: n_dims.len() as u32,
                shape: n_dims,
                quant_type: QuantType::F32,
                offset: data_off,
            });
            data_off += data.len() as u64;
        }

        // ─── 3. Build KV metadata ───
        let kv = vec![
            GgufKv { key: "general.architecture".into(), value: GgufValue::String("amadeus".into()) },
            GgufKv { key: "amadeus.format_version".into(), value: GgufValue::U32(FORMAT_VERSION) },
            GgufKv { key: "amadeus.syntax_order".into(), value: GgufValue::U32(self.syntax_order as u32) },
            GgufKv { key: "amadeus.temperature".into(), value: GgufValue::F32(self.temperature) },
            GgufKv { key: "amadeus.exploration_rate".into(), value: GgufValue::F32(self.exploration_rate) },
            GgufKv { key: "amadeus.graph_alpha".into(), value: GgufValue::F32(self.graph_alpha) },
        ];

        // ─── 4. Write header + data (no alignment padding in data) ───
        let mut w = GgufWriter::new();
        w.write_header(3, n_tensors, kv.len() as u64, &kv, &tensor_infos);

        // Write all sections directly (no padding)
        for (_name, data) in &sections {
            w.write_raw(data);
        }

        let bytes = w.finish();
        std::fs::write(path, &bytes)
    }

    /// Load grammar from GGUF format.
    pub fn load_gguf(&mut self, path: &str) -> std::io::Result<()> {
        use crate::quant::gguf::{parse_header, GgufValue};

        let data = std::fs::read(path)?;
        let header = parse_header(&data).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        // Verify architecture
        let arch = header.kv_pairs.iter().find(|kv| kv.key == "general.architecture")
            .and_then(|kv| if let GgufValue::String(s) = &kv.value { Some(s.as_str()) } else { None });
        if arch != Some("amadeus") {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "not an amadeus GGUF file"));
        }

        // Read metadata
        let get_u32 = |key: &str| -> Option<u32> {
            header.kv_pairs.iter().find(|kv| kv.key == key).and_then(|kv| match &kv.value {
                GgufValue::U32(v) => Some(*v),
                GgufValue::U64(v) => Some(*v as u32),
                _ => None,
            })
        };
        let get_f32 = |key: &str| -> Option<f32> {
            header.kv_pairs.iter().find(|kv| kv.key == key).and_then(|kv| match &kv.value {
                GgufValue::F32(v) => Some(*v),
                GgufValue::F64(v) => Some(*v as f32),
                _ => None,
            })
        };

        if let Some(v) = get_u32("amadeus.syntax_order") { self.syntax_order = v as usize; }
        if let Some(v) = get_f32("amadeus.temperature") { self.temperature = v; }
        if let Some(v) = get_f32("amadeus.exploration_rate") { self.exploration_rate = v; }
        if let Some(v) = get_f32("amadeus.graph_alpha") { self.graph_alpha = v; }

        // Build tensor name → info map
        let tensor_map: std::collections::HashMap<&str, &crate::quant::gguf::GgufTensorInfo> = header
            .tensor_infos.iter().map(|t| (t.name.as_str(), t)).collect();

        // Helper: read raw tensor bytes
        let read_tensor = |name: &str| -> Option<&[u8]> {
            let info = tensor_map.get(name)?;
            let start = header.data_offset as usize + info.offset as usize;
            let numel: usize = info.shape.iter().map(|&d| d as usize).product();
            let byte_size = numel * 4;
            let end = start + byte_size;
            if end > data.len() {
                eprintln!("  WARN: tensor '{}' end {} > file {}", name, end, data.len());
                return None;
            }
            Some(&data[start..end])
        };

        // Load GRAPH
        if let Some(raw) = read_tensor("graph.data") {
            self.graph.clear();
            for chunk in raw.chunks_exact(8) {
                let lex_id = u32::from_le_bytes(chunk[0..4].try_into().unwrap());
                let sig = u32::from_le_bytes(chunk[4..8].try_into().unwrap());
                self.graph.insert(lex_id, sig);
            }
        }

        // Load lex_to_class
        if let Some(raw) = read_tensor("lex_to_class.data") {
            self.lex_to_class.clear();
            for chunk in raw.chunks_exact(8) {
                let lex_id = u32::from_le_bytes(chunk[0..4].try_into().unwrap());
                let cls = u32::from_le_bytes(chunk[4..8].try_into().unwrap()) as u8;
                self.lex_to_class.insert(lex_id, cls);
            }
        }

        // Load T2 head
        if let Some(raw) = read_tensor("t2.head.data") {
            self.agreement.clear();
            self.agreement_totals.clear();
            Self::read_t2_flat(raw, &mut self.agreement, &mut self.agreement_totals);
        }
        // Load T2 gparent
        if let Some(raw) = read_tensor("t2.gparent.data") {
            self.agreement_gparent.clear();
            self.agreement_gparent_totals.clear();
            Self::read_t2_flat(raw, &mut self.agreement_gparent, &mut self.agreement_gparent_totals);
        }
        // Load T2 class
        if let Some(raw) = read_tensor("t2.class.data") {
            self.agreement_class.clear();
            self.agreement_class_totals.clear();
            Self::read_t2_class_flat(raw, &mut self.agreement_class, &mut self.agreement_class_totals);
        }

        // Load T3 exact
        if let Some(raw) = read_tensor("t3.exact.data") {
            self.lexicon.clear();
            self.lexicon_totals.clear();
            Self::read_t3_flat(raw, &mut self.lexicon, &mut self.lexicon_totals);
        }
        // Load T3 CSS
        if let Some(raw) = read_tensor("t3.css.data") {
            self.lexicon_cls_syn_style.clear();
            self.lexicon_cls_syn_style_totals.clear();
            Self::read_t3_css_flat(raw, &mut self.lexicon_cls_syn_style, &mut self.lexicon_cls_syn_style_totals);
        }
        // Load T3 CS
        if let Some(raw) = read_tensor("t3.cs.data") {
            self.lexicon_cls_syn.clear();
            self.lexicon_cls_syn_totals.clear();
            Self::read_t3_cs_flat(raw, &mut self.lexicon_cls_syn, &mut self.lexicon_cls_syn_totals);
        }
        // Load T3 CST
        if let Some(raw) = read_tensor("t3.cst.data") {
            self.lexicon_cls_style.clear();
            self.lexicon_cls_style_totals.clear();
            Self::read_t3_cst_flat(raw, &mut self.lexicon_cls_style, &mut self.lexicon_cls_style_totals);
        }
        // Load T3 class
        if let Some(raw) = read_tensor("t3.class.data") {
            self.lexicon_class.clear();
            self.lexicon_class_totals.clear();
            Self::read_t3_class_flat(raw, &mut self.lexicon_class, &mut self.lexicon_class_totals);
        }

        // Load CUBO blobs
        if let Some(raw) = read_tensor("cubo.clause.blob") {
            Self::read_cubo_blob(raw, &mut self.hier.clause);
        }
        if let Some(raw) = read_tensor("cubo.sentence.blob") {
            Self::read_cubo_blob(raw, &mut self.hier.sentence);
        }
        if let Some(raw) = read_tensor("cubo.paragraph.blob") {
            Self::read_cubo_blob(raw, &mut self.hier.paragraph);
        }
        if let Some(raw) = read_tensor("cubo.text.blob") {
            Self::read_cubo_blob(raw, &mut self.hier.text);
        }

        // Load SNN weights (se disponível)
        if let (Some(raw_ih), Some(raw_ho), Some(raw_labels)) = (
            read_tensor("snn.synapses_ih"),
            read_tensor("snn.synapses_ho"),
            read_tensor("snn.output_labels"),
        ) {
            use crate::amadeus_m::snn::SpikingNetwork;

            // Converter bytes → f32 slices
            let labels_f32: &[f32] = unsafe { std::slice::from_raw_parts(raw_labels.as_ptr() as *const f32, raw_labels.len() / 4) };
            let labels = SpikingNetwork::from_flat_labels(labels_f32);
            let n_out = labels.len();

            // Determinar dimensões dos tensores
            let ih_info = tensor_map.get("snn.synapses_ih").unwrap();
            let n_input = ih_info.shape[0] as usize;
            let n_hidden = ih_info.shape[1] as usize;

            let ih_f32: &[f32] = unsafe { std::slice::from_raw_parts(raw_ih.as_ptr() as *const f32, raw_ih.len() / 4) };
            let ho_f32: &[f32] = unsafe { std::slice::from_raw_parts(raw_ho.as_ptr() as *const f32, raw_ho.len() / 4) };
            let synapses_ih = SpikingNetwork::from_flat_ih(ih_f32, n_input, n_hidden);
            let synapses_ho = SpikingNetwork::from_flat_ho(ho_f32, n_hidden, n_out);

            // Reconstruir rede com pesos carregados
            let tau = 5.0;
            let threshold = 1.0;
            let refrac = 2;
            let dt = 1.0;
            let tsteps = 16;

            let mut net = SpikingNetwork::new(n_input, n_hidden, labels, tau, threshold, refrac, dt, tsteps);
            net.synapses_ih = synapses_ih;
            net.synapses_ho = synapses_ho;

            self.snn = Some(net);
            self.snn_enc = Some(crate::amadeus_m::snn::MorphEncoding::new());
        }

        Ok(())
    }

    // ─── GGUF flat serialization helpers ───

    fn serialize_t2_flat(map: &HashMap<(u16, u16, u8), HashMap<(u16, u16), f32>>, totals: &HashMap<(u16, u16, u8), f32>) -> Vec<u8> {
        let mut buf = Vec::new();
        for (&(m1, s1, c), nexts) in map {
            let total = totals.get(&(m1, s1, c)).copied().unwrap_or(0.0);
            for (&(m2, s2), &v) in nexts {
                buf.extend_from_slice(&(m1 as u32).to_le_bytes());
                buf.extend_from_slice(&(s1 as u32).to_le_bytes());
                buf.extend_from_slice(&(c as u32).to_le_bytes());
                buf.extend_from_slice(&(m2 as u32).to_le_bytes());
                buf.extend_from_slice(&(s2 as u32).to_le_bytes());
                buf.extend_from_slice(&v.to_le_bytes());
            }
        }
        buf
    }

    fn serialize_t2_class_flat(map: &HashMap<u8, HashMap<(u16, u16), f32>>, totals: &HashMap<u8, f32>) -> Vec<u8> {
        let mut buf = Vec::new();
        for (&c, nexts) in map {
            let total = totals.get(&c).copied().unwrap_or(0.0);
            for (&(m2, s2), &v) in nexts {
                buf.extend_from_slice(&(c as u32).to_le_bytes());
                buf.extend_from_slice(&(m2 as u32).to_le_bytes());
                buf.extend_from_slice(&(s2 as u32).to_le_bytes());
                buf.extend_from_slice(&v.to_le_bytes());
            }
        }
        buf
    }

    fn serialize_t3_flat(map: &HashMap<(u16, u8, u16), HashMap<u32, f32>>, totals: &HashMap<(u16, u8, u16), f32>) -> Vec<u8> {
        let mut buf = Vec::new();
        for (&(m, sf, st), nexts) in map {
            let total = totals.get(&(m, sf, st)).copied().unwrap_or(0.0);
            for (&id, &v) in nexts {
                buf.extend_from_slice(&(m as u32).to_le_bytes());
                buf.extend_from_slice(&(sf as u32).to_le_bytes());
                buf.extend_from_slice(&(st as u32).to_le_bytes());
                buf.extend_from_slice(&id.to_le_bytes());
                buf.extend_from_slice(&v.to_le_bytes());
            }
        }
        buf
    }

    fn serialize_t3_css_flat(map: &HashMap<(u8, u8, u16), HashMap<u32, f32>>, totals: &HashMap<(u8, u8, u16), f32>) -> Vec<u8> {
        let mut buf = Vec::new();
        for (&(c, sf, st), nexts) in map {
            let total = totals.get(&(c, sf, st)).copied().unwrap_or(0.0);
            for (&id, &v) in nexts {
                buf.extend_from_slice(&(c as u32).to_le_bytes());
                buf.extend_from_slice(&(sf as u32).to_le_bytes());
                buf.extend_from_slice(&(st as u32).to_le_bytes());
                buf.extend_from_slice(&id.to_le_bytes());
                buf.extend_from_slice(&v.to_le_bytes());
            }
        }
        buf
    }

    fn serialize_t3_cs_flat(map: &HashMap<(u8, u8), HashMap<u32, f32>>, totals: &HashMap<(u8, u8), f32>) -> Vec<u8> {
        let mut buf = Vec::new();
        for (&(c, sf), nexts) in map {
            let total = totals.get(&(c, sf)).copied().unwrap_or(0.0);
            for (&id, &v) in nexts {
                buf.extend_from_slice(&(c as u32).to_le_bytes());
                buf.extend_from_slice(&(sf as u32).to_le_bytes());
                buf.extend_from_slice(&id.to_le_bytes());
                buf.extend_from_slice(&v.to_le_bytes());
            }
        }
        buf
    }

    fn serialize_t3_cst_flat(map: &HashMap<(u8, u16), HashMap<u32, f32>>, totals: &HashMap<(u8, u16), f32>) -> Vec<u8> {
        let mut buf = Vec::new();
        for (&(c, st), nexts) in map {
            let total = totals.get(&(c, st)).copied().unwrap_or(0.0);
            for (&id, &v) in nexts {
                buf.extend_from_slice(&(c as u32).to_le_bytes());
                buf.extend_from_slice(&(st as u32).to_le_bytes());
                buf.extend_from_slice(&id.to_le_bytes());
                buf.extend_from_slice(&v.to_le_bytes());
            }
        }
        buf
    }

    fn serialize_t3_class_flat(map: &HashMap<u8, HashMap<u32, f32>>, totals: &HashMap<u8, f32>) -> Vec<u8> {
        let mut buf = Vec::new();
        for (&c, nexts) in map {
            let total = totals.get(&c).copied().unwrap_or(0.0);
            for (&id, &v) in nexts {
                buf.extend_from_slice(&(c as u32).to_le_bytes());
                buf.extend_from_slice(&id.to_le_bytes());
                buf.extend_from_slice(&v.to_le_bytes());
            }
        }
        buf
    }

    fn serialize_cubo_blob(cube: &crate::amadeus_m::hypercube::HyperCube) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&(cube.table.len() as u32).to_le_bytes());
        for (ctx, nexts) in &cube.table {
            buf.extend_from_slice(&(ctx.len() as u32).to_le_bytes());
            for &v in ctx { buf.extend_from_slice(&v.to_le_bytes()); }
            buf.extend_from_slice(&(nexts.len() as u32).to_le_bytes());
            for (&k, &v) in nexts {
                buf.extend_from_slice(&k.to_le_bytes());
                buf.extend_from_slice(&v.to_le_bytes());
            }
        }
        buf.extend_from_slice(&(cube.unigram.len() as u32).to_le_bytes());
        buf.extend_from_slice(&cube.total_lex.to_le_bytes());
        for (&k, &v) in &cube.unigram {
            buf.extend_from_slice(&k.to_le_bytes());
            buf.extend_from_slice(&v.to_le_bytes());
        }
        buf.extend_from_slice(&(cube.lambda.len() as u32).to_le_bytes());
        for &l in &cube.lambda { buf.extend_from_slice(&l.to_le_bytes()); }
        buf
    }

    fn read_cubo_blob(raw: &[u8], cube: &mut crate::amadeus_m::hypercube::HyperCube) {
        let mut off = 0;
        cube.table.clear();
        cube.totals.clear();
        cube.unigram.clear();
        cube.total_lex = 0.0;

        if raw.len() < 4 { return; }

        // Read n_contexts
        let n_contexts = u32::from_le_bytes(raw[off..off+4].try_into().unwrap()) as usize;
        off += 4;

        // Read context entries
        let order = cube.order;
        for _ in 0..n_contexts {
            if off + 4 > raw.len() { break; }
            let ctx_len = u32::from_le_bytes(raw[off..off+4].try_into().unwrap()) as usize;
            off += 4;
            if ctx_len > order + 1 || ctx_len == 0 { break; }
            if off + ctx_len * 16 > raw.len() { break; }
            let mut ctx = Vec::with_capacity(ctx_len);
            for _ in 0..ctx_len {
                let v = u128::from_le_bytes(raw[off..off+16].try_into().unwrap());
                off += 16;
                ctx.push(v);
            }
            if off + 4 > raw.len() { break; }
            let n_next = u32::from_le_bytes(raw[off..off+4].try_into().unwrap()) as usize;
            off += 4;
            if off + n_next * 8 > raw.len() { break; }
            let mut nexts = HashMap::with_capacity(n_next);
            let mut sum = 0.0f32;
            for _ in 0..n_next {
                let k = u32::from_le_bytes(raw[off..off+4].try_into().unwrap()); off += 4;
                let v = f32::from_le_bytes(raw[off..off+4].try_into().unwrap()); off += 4;
                sum += v;
                nexts.insert(k, v);
            }
            cube.totals.insert(ctx.clone(), sum);
            cube.table.insert(ctx, nexts);
        }

        // Read unigram section
        if off + 4 <= raw.len() {
            let n_uni = u32::from_le_bytes(raw[off..off+4].try_into().unwrap()) as usize;
            off += 4;
            if off + 4 <= raw.len() {
                cube.total_lex = f32::from_le_bytes(raw[off..off+4].try_into().unwrap());
                off += 4;
                for _ in 0..n_uni {
                    if off + 8 > raw.len() { break; }
                    let k = u32::from_le_bytes(raw[off..off+4].try_into().unwrap()); off += 4;
                    let v = f32::from_le_bytes(raw[off..off+4].try_into().unwrap()); off += 4;
                    cube.unigram.insert(k, v);
                }
            }
        }

        // Read lambda section
        if off + 4 <= raw.len() {
            let n_lam = u32::from_le_bytes(raw[off..off+4].try_into().unwrap()) as usize;
            off += 4;
            cube.lambda.clear();
            for _ in 0..n_lam {
                if off + 4 > raw.len() { break; }
                let l = f32::from_le_bytes(raw[off..off+4].try_into().unwrap());
                off += 4;
                cube.lambda.push(l);
            }
        }
    }

    fn read_t2_flat(raw: &[u8], map: &mut HashMap<(u16, u16, u8), HashMap<(u16, u16), f32>>, totals: &mut HashMap<(u16, u16, u8), f32>) {
        for chunk in raw.chunks_exact(24) { // 6 * u32 = 24 bytes
            let m1 = u32::from_le_bytes(chunk[0..4].try_into().unwrap()) as u16;
            let s1 = u32::from_le_bytes(chunk[4..8].try_into().unwrap()) as u16;
            let c = u32::from_le_bytes(chunk[8..12].try_into().unwrap()) as u8;
            let m2 = u32::from_le_bytes(chunk[12..16].try_into().unwrap()) as u16;
            let s2 = u32::from_le_bytes(chunk[16..20].try_into().unwrap()) as u16;
            let v = f32::from_le_bytes(chunk[20..24].try_into().unwrap());
            *map.entry((m1, s1, c)).or_default().entry((m2, s2)).or_insert(0.0) += v;
            *totals.entry((m1, s1, c)).or_insert(0.0) += v;
        }
    }

    fn read_t2_class_flat(raw: &[u8], map: &mut HashMap<u8, HashMap<(u16, u16), f32>>, totals: &mut HashMap<u8, f32>) {
        for chunk in raw.chunks_exact(16) { // 4 * u32 = 16 bytes
            let c = u32::from_le_bytes(chunk[0..4].try_into().unwrap()) as u8;
            let m2 = u32::from_le_bytes(chunk[4..8].try_into().unwrap()) as u16;
            let s2 = u32::from_le_bytes(chunk[8..12].try_into().unwrap()) as u16;
            let v = f32::from_le_bytes(chunk[12..16].try_into().unwrap());
            *map.entry(c).or_default().entry((m2, s2)).or_insert(0.0) += v;
            *totals.entry(c).or_insert(0.0) += v;
        }
    }

    fn read_t3_flat(raw: &[u8], map: &mut HashMap<(u16, u8, u16), HashMap<u32, f32>>, totals: &mut HashMap<(u16, u8, u16), f32>) {
        for chunk in raw.chunks_exact(20) { // 5 * u32 = 20 bytes
            let m = u32::from_le_bytes(chunk[0..4].try_into().unwrap()) as u16;
            let sf = u32::from_le_bytes(chunk[4..8].try_into().unwrap()) as u8;
            let st = u32::from_le_bytes(chunk[8..12].try_into().unwrap()) as u16;
            let id = u32::from_le_bytes(chunk[12..16].try_into().unwrap());
            let v = f32::from_le_bytes(chunk[16..20].try_into().unwrap());
            *map.entry((m, sf, st)).or_default().entry(id).or_insert(0.0) += v;
            *totals.entry((m, sf, st)).or_insert(0.0) += v;
        }
    }

    fn read_t3_css_flat(raw: &[u8], map: &mut HashMap<(u8, u8, u16), HashMap<u32, f32>>, totals: &mut HashMap<(u8, u8, u16), f32>) {
        for chunk in raw.chunks_exact(20) { // 5 * u32 = 20 bytes
            let c = u32::from_le_bytes(chunk[0..4].try_into().unwrap()) as u8;
            let sf = u32::from_le_bytes(chunk[4..8].try_into().unwrap()) as u8;
            let st = u32::from_le_bytes(chunk[8..12].try_into().unwrap()) as u16;
            let id = u32::from_le_bytes(chunk[12..16].try_into().unwrap());
            let v = f32::from_le_bytes(chunk[16..20].try_into().unwrap());
            *map.entry((c, sf, st)).or_default().entry(id).or_insert(0.0) += v;
            *totals.entry((c, sf, st)).or_insert(0.0) += v;
        }
    }

    fn read_t3_cs_flat(raw: &[u8], map: &mut HashMap<(u8, u8), HashMap<u32, f32>>, totals: &mut HashMap<(u8, u8), f32>) {
        for chunk in raw.chunks_exact(16) { // 4 * u32 = 16 bytes
            let c = u32::from_le_bytes(chunk[0..4].try_into().unwrap()) as u8;
            let sf = u32::from_le_bytes(chunk[4..8].try_into().unwrap()) as u8;
            let id = u32::from_le_bytes(chunk[8..12].try_into().unwrap());
            let v = f32::from_le_bytes(chunk[12..16].try_into().unwrap());
            *map.entry((c, sf)).or_default().entry(id).or_insert(0.0) += v;
            *totals.entry((c, sf)).or_insert(0.0) += v;
        }
    }

    fn read_t3_cst_flat(raw: &[u8], map: &mut HashMap<(u8, u16), HashMap<u32, f32>>, totals: &mut HashMap<(u8, u16), f32>) {
        for chunk in raw.chunks_exact(16) { // 4 * u32 = 16 bytes
            let c = u32::from_le_bytes(chunk[0..4].try_into().unwrap()) as u8;
            let st = u32::from_le_bytes(chunk[4..8].try_into().unwrap()) as u16;
            let id = u32::from_le_bytes(chunk[8..12].try_into().unwrap());
            let v = f32::from_le_bytes(chunk[12..16].try_into().unwrap());
            *map.entry((c, st)).or_default().entry(id).or_insert(0.0) += v;
            *totals.entry((c, st)).or_insert(0.0) += v;
        }
    }

    fn read_t3_class_flat(raw: &[u8], map: &mut HashMap<u8, HashMap<u32, f32>>, totals: &mut HashMap<u8, f32>) {
        for chunk in raw.chunks_exact(12) { // 3 * u32 = 12 bytes
            let c = u32::from_le_bytes(chunk[0..4].try_into().unwrap()) as u8;
            let id = u32::from_le_bytes(chunk[4..8].try_into().unwrap());
            let v = f32::from_le_bytes(chunk[8..12].try_into().unwrap());
            *map.entry(c).or_default().entry(id).or_insert(0.0) += v;
            *totals.entry(c).or_insert(0.0) += v;
        }
    }
}
