use std::collections::HashMap;
use crate::amadeus_m::compact::{chave, Cands, CtxKey, FxHashMap};
use crate::amadeus_m::token7::Token7;

// ─── 8D packing: cada token de histórico → u128 ───
//
// Bits baixos (u64): LEX_HASH(16) | MORPH(16) | SYN_BIN(4) | SYN_FUNC(4) | PUNCT(4) | STYLE(8) | CLAUSE(4) | RSVD(8)
// Bits altos (u32): GRAPH(32)
// Bits altos restantes: RESERVED(32)

const SH_LEX:    usize = 0;
const SH_MORPH:  usize = 16;
const SH_SYN_BIN: usize = 32;
const SH_SYN_FUNC: usize = 36;
const SH_PUNCT:  usize = 40;
const SH_STYLE:  usize = 44;
const SH_CLAUSE: usize = 52;
const SH_GRAPH:  usize = 64;

pub fn pack8d(t: &Token7, clause_depth: u8) -> u128 {
    let lex_hash = (t.lex ^ (t.lex >> 16)) as u128 & 0xFFFF;
    let syn_bin = syn_bin(t.syn_off);
    (lex_hash << SH_LEX)
        | ((t.morph as u128) << SH_MORPH)
        | ((syn_bin as u128) << SH_SYN_BIN)
        | ((t.syn_func as u128) << SH_SYN_FUNC)
        | ((t.punct as u128) << SH_PUNCT)
        | ((t.style as u128) << SH_STYLE)
        | ((clause_depth as u128) << SH_CLAUSE)
        | ((t.graph as u128) << SH_GRAPH)
}

fn syn_bin(off: i16) -> u8 {
    if off == 0 { 0 }
    else if off >= -1 { 1 }
    else if off >= -3 { 2 }
    else if off >= -7 { 3 }
    else { 4 }
}

// ─── HyperCube ───

pub struct HyperCube {
    pub order: usize,
    // Chave canônica u64 em vez de Vec<u128>, e Cands (Vec ordenado) em vez
    // de HashMap interno. Ver `compact.rs` para a medição: 266 → 182 bytes por
    // entrada, e FxHash em vez de SipHash.
    pub table: FxHashMap<CtxKey, Cands>,
    pub totals: FxHashMap<CtxKey, f32>,
    // unigram
    pub unigram: FxHashMap<u32, f32>,
    pub total_lex: f32,
    // pesos de interpolação fixos
    pub lambda: Vec<f32>,
    unigram_weight: f32,
    /// Store em disco quando a tabela é grande demais para RAM. Ver `shard.rs`.
    /// Quando presente, `sample_*` lê daqui e `table` fica vazia.
    pub shards: Option<crate::amadeus_m::shard::ShardStore>,
    /// Se true, `train()` faz flush periódico em vez de crescer em RAM.
    pub spill_enabled: bool,
    /// Contextos acumulados desde o último flush (modo spill).
    pub pending: FxHashMap<CtxKey, Cands>,
    pending_totals: FxHashMap<CtxKey, f32>,
    /// A cada quantos contextos acumulados acontece o flush.
    pub spill_threshold: usize,
}

impl HyperCube {
    pub fn new(order: usize) -> Self {
        let lambda: Vec<f32> = (0..order).map(|i| 0.5_f32.powi(i as i32 + 1)).collect();
        let sum_l: f32 = lambda.iter().sum();
        let lambda: Vec<f32> = lambda.into_iter().map(|l| l / sum_l).collect();
        let unigram_weight = 0.10;
        Self {
            order,
            table: FxHashMap::default(),
            totals: FxHashMap::default(),
            unigram: FxHashMap::default(),
            total_lex: 0.0,
            lambda,
            unigram_weight,
            shards: None,
            spill_enabled: false,
            pending: FxHashMap::default(),
            pending_totals: FxHashMap::default(),
            spill_threshold: 2_000_000,
        }
    }

    /// Habilita spill para disco. A partir daqui `train()` acumula em
    /// `pending` e descarrega em shards, em vez de crescer `table` em RAM.
    pub fn enable_spill(&mut self, dir: impl AsRef<std::path::Path>) -> std::io::Result<()> {
        self.shards = Some(crate::amadeus_m::shard::ShardStore::open(dir)?);
        self.spill_enabled = true;
        Ok(())
    }

    /// Carrega um modelo shardado existente. Deixa `table` vazia: a leitura
    /// passa a vir do mmap.
    pub fn load_shards(&mut self, dir: impl AsRef<std::path::Path>) -> std::io::Result<usize> {
        let store = crate::amadeus_m::shard::ShardStore::open_dir(dir)?;
        let n = store.total_contexts();
        self.shards = Some(store);
        self.table.clear();
        self.totals.clear();
        Ok(n)
    }

    /// Migra a tabela em RAM para `pending` e habilita spill. Usado para
    /// converter um modelo já carregado (ex.: via `load_gguf`) em um
    /// shardado, sem precisar retreinar.
    ///
    /// Chame `flush_pending()` depois para materializar o primeiro shard.
    pub fn migrate_to_spill(&mut self, dir: impl AsRef<std::path::Path>) -> std::io::Result<()> {
        self.pending = std::mem::take(&mut self.table);
        self.pending_totals = std::mem::take(&mut self.totals);
        self.enable_spill(dir)
    }

    /// Descarrega os contextos pendentes para um shard. Sem efeito se vazio.
    pub fn flush_pending(&mut self) -> std::io::Result<Option<std::path::PathBuf>> {
        if self.pending.is_empty() {
            return Ok(None);
        }
        let Some(store) = self.shards.as_mut() else {
            return Ok(None);
        };
        let path = store.flush(&self.pending, &self.pending_totals, self.total_lex)?;
        // `clear()` mantém a capacidade alocada — os buckets ficariam
        // reservados (GB) e a memória não cairia. `take` devolve um mapa vazio
        // e a alocação é devolvida ao allocator.
        self.pending = FxHashMap::default();
        self.pending_totals = FxHashMap::default();
        Ok(Some(path))
    }

    /// Busca um contexto: `pending` primeiro (dados mais recentes, somados aos
    /// antigos), depois shards, depois `table`.
    ///
    /// A ordem importa: um contexto pode existir num shard antigo e ter sido
    /// retreinado depois. Se o shard ganhasse, o treino recente seria
    /// silenciosamente ignorado.
    /// Retorna (candidatos, soma).
    pub fn lookup_ctx(&mut self, ctx: &[u128]) -> Option<(Cands, f32)> {
        let k = chave(ctx);
        if let Some(cands) = self.pending.get(&k) {
            // Some as contagens do shard antigo, para manter o histórico.
            let mut merged = cands.clone();
            if let Some(store) = self.shards.as_mut() {
                if let Some((old, _)) = store.get(ctx) {
                    merged.merge(&old, 1.0);
                }
            } else if let Some(old) = self.table.get(&k) {
                merged.merge(old, 1.0);
            }
            let sum = merged.total();
            return Some((merged, sum));
        }

        if let Some(store) = self.shards.as_mut() {
            if let Some(found) = store.get(ctx) {
                return Some(found);
            }
        }

        self.table.get(&k).map(|c| {
            let sum = self.totals.get(&k).copied().unwrap_or_else(|| c.total());
            (c.clone(), sum)
        })
    }

    pub fn train(&mut self, tokens: &[Token7], clause_depths: &[u8]) {
        if tokens.is_empty() { return; }
        // Pre-computar contextos para treino EM posterior
        let mut ctx_cache: Vec<Vec<Vec<u128>>> = Vec::with_capacity(tokens.len());
        for i in 0..tokens.len() {
            let max_n = self.order.min(i);
            let mut ctxs = Vec::with_capacity(max_n);
            for n in 1..=max_n {
                let ctx: Vec<u128> = (i - n..i)
                    .map(|j| pack8d(&tokens[j], clause_depths.get(j).copied().unwrap_or(0)))
                    .collect();
                ctxs.push(ctx);
            }
            ctx_cache.push(ctxs);
        }
        for i in 0..tokens.len() {
            let t = &tokens[i];
            let lex = t.lex;
            // Log-frequency para reduzir domínio de palavras frequentes
            let count = self.unigram.entry(lex).or_insert(0.0);
            *count += 1.0;
            *count = count.ln_1p(); // ln(1 + count) suaviza
            self.total_lex += 1.0;

            let max_n = self.order.min(i);
            for n in 0..max_n {
                let ctx = &ctx_cache[i][n];
                let k = chave(ctx);
                if self.spill_enabled {
                    self.pending.entry(k).or_default().add(lex, 1.0);
                    *self.pending_totals.entry(k).or_insert(0.0) += 1.0;
                } else {
                    self.table.entry(k).or_default().add(lex, 1.0);
                    *self.totals.entry(k).or_insert(0.0) += 1.0;
                }
            }
        }

        // EM: aprender lambdas dos dados
        self.em_train_lambdas(&tokens, &ctx_cache);

        // Descarrega quando o lote pendente passa do limite. Sem isso, `pending`
        // cresceria sem teto e o ganho de memória se perderia.
        if self.spill_enabled && self.pending.len() >= self.spill_threshold {
            let _ = self.flush_pending();
        }
    }

    fn em_train_lambdas(&mut self, tokens: &[Token7], ctx_cache: &[Vec<Vec<u128>>]) {
        // EM com piso mínimo — garante que ordens superiores mantenham peso mínimo
        // mesmo sendo esparsas. O piso é exponencial decrescente.
        let order = self.order;
        if order == 0 || tokens.len() < 2 { return; }

        // Piso mínimo para cada ordem: [0.20, 0.10, 0.05, ...]
        let floor: Vec<f64> = (0..order).map(|i| 0.20_f64 * 0.5_f64.powi(i as i32)).collect();
        let mut gamma_sum = vec![0.0f64; order];
        let smooth = 1e-10;

        for i in 1..tokens.len() {
            let lex = tokens[i].lex;
            let max_n = order.min(i);
            if max_n == 0 { continue; }

            let mut probs = vec![0.0f64; order];
            for n in 0..max_n {
                // Via lookup_ctx: no modo spill os contadores estão em `pending`
                // ou em shards, não em `table`. Ler `table` direto daria
                // probs=0 e deixaria os lambdas no valor inicial.
                if let Some((cands, total)) = self.lookup_ctx(&ctx_cache[i][n]) {
                    let total = total.max(1.0);
                    probs[n] = (cands.get(lex).unwrap_or(0.0) / total) as f64;
                }
            }

            let pu = if self.total_lex > 0.0 {
                (self.unigram.get(&lex).copied().unwrap_or(0.0) / self.total_lex) as f64
            } else { 0.0 };

            let mut numerator = 0.0f64;
            for n in 0..order {
                numerator += self.lambda[n] as f64 * probs[n];
            }
            numerator += self.unigram_weight as f64 * pu;
            numerator = numerator.max(smooth);

            for n in 0..order {
                if probs[n] > 0.0 {
                    let contrib = self.lambda[n] as f64 * probs[n];
                    gamma_sum[n] += contrib / numerator;
                }
            }
        }

        let sum_g: f64 = gamma_sum.iter().sum();
        if sum_g > 0.0 {
            // Interpolar EM (data-driven) com piso (prior): λ_final = 0.7 * λ_em + 0.3 * λ_floor
            let floor_sum: f64 = floor.iter().sum();
            for n in 0..order {
                let em_l = gamma_sum[n] / sum_g;
                let floor_l = floor[n] / floor_sum;
                self.lambda[n] = (0.7 * em_l + 0.3 * floor_l) as f32;
            }
        }
    }

    pub fn sample_lex(&mut self, history: &[Token7], clause_depths: &[u8], temp: f32, explore: f32, rng: &mut fastrand::Rng) -> u32 {
        self.sample_lex_modulated(history, clause_depths, temp, explore, rng, None, 0.0)
    }

    pub fn sample_lex_modulated(&mut self, history: &[Token7], clause_depths: &[u8], temp: f32, explore: f32, rng: &mut fastrand::Rng, graph: Option<&HashMap<u32, u32>>, alpha: f32) -> u32 {
        let mut dist: HashMap<u32, f32> = HashMap::new();

        let max_n = self.order.min(history.len());
        for n in 1..=max_n {
            let ctx: Vec<u128> = (history.len() - n..history.len())
                .map(|j| pack8d(&history[j], clause_depths.get(j).copied().unwrap_or(0)))
                .collect();
            if let Some((cands, total)) = self.lookup_ctx(&ctx) {
                if total > 0.0 {
                    let w = self.lambda.get(n - 1).copied().unwrap_or(0.1);
                    for &(lx, cnt) in &cands.items {
                        *dist.entry(lx).or_insert(0.0) += w * cnt / total;
                    }
                }
            }
        }

        if self.total_lex > 0.0 {
            let uw = self.unigram_weight;
            for (&lx, &cnt) in &self.unigram {
                *dist.entry(lx).or_insert(0.0) += uw * cnt / self.total_lex;
            }
        }

        // Modulação por GRAPH: espalhar massa entre palavras semanticamente similares
        if let Some(g) = graph {
            if alpha > 0.0 && dist.len() >= 2 {
                let top_k = 20usize.min(dist.len());
                let mut items: Vec<(u32, f32)> = dist.iter().map(|(&k, &v)| (k, v)).collect();
                // Desempate por lexema: sort_by é instável e a ordem de
                // iteração do HashMap varia com o layout.
                items.sort_by(|a, b| {
                    b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then(a.0.cmp(&b.0))
                });
                let spread: Vec<(u32, f32)> = items.iter().take(top_k).copied().collect();
                let mut added: HashMap<u32, f32> = HashMap::new();
                for &(lex, prob) in &spread {
                    let g_self = g.get(&lex).copied().unwrap_or(0);
                    let mut total_sim = 0.0f32;
                    for &(other, _) in &spread {
                        if other == lex { continue; }
                        let g_other = g.get(&other).copied().unwrap_or(0);
                        let dist_ham = (g_self ^ g_other).count_ones() as f32;
                        let sim = 1.0 - dist_ham / 32.0;
                        if sim > 0.5 {
                            total_sim += sim;
                        }
                    }
                    if total_sim > 0.0 {
                        let boost = prob * alpha * total_sim / spread.len() as f32;
                        for &(other, _) in &spread {
                            if other == lex { continue; }
                            let g_other = g.get(&other).copied().unwrap_or(0);
                            let sim = 1.0 - (g_self ^ g_other).count_ones() as f32 / 32.0;
                            if sim > 0.5 {
                                *added.entry(other).or_insert(0.0) += boost * sim / total_sim;
                            }
                        }
                    }
                }
                let mut pares: Vec<(u32, f32)> = added.into_iter().collect();
                pares.sort_unstable_by_key(|&(k, _)| k);
                for (other, boost) in pares {
                    *dist.entry(other).or_insert(0.0) += boost;
                }
            }
        }

        if dist.is_empty() { return 0; }

        if rng.f32() < explore {
            // Ordena por lexema: a ordem de iteração de um HashMap depende do
            // layout, que muda entre construção por `entry()` e por
            // `with_capacity`. Sem ordenar, a mesma semente dá tokens diferentes.
            let mut keys: Vec<u32> = dist.keys().copied().filter(|&k| k != 0).collect();
            keys.sort_unstable();
            if !keys.is_empty() {
                return keys[rng.usize(0..keys.len())];
            }
            let mut any: Vec<u32> = dist.keys().copied().collect();
            any.sort_unstable();
            return any.first().copied().unwrap_or(0);
        }

        let t = temp.max(0.01);
        let mut sum = 0.0f64;
        let mut probs: Vec<(u32, f64)> = Vec::new();
        for (&lx, &prob) in &dist {
            let w = (prob as f64).powf(1.0 / t as f64);
            if w > 0.0 { sum += w; probs.push((lx, w)); }
        }
        if sum <= 0.0 { return 0; }

        // Ordena por lexema antes da varredura cumulativa. A ordem de iteração
        // do HashMap é detalhe de implementação, e a amostra depende dela:
        // sem isso, dois CUBOs com o mesmo conteúdo geram sequências distintas.
        probs.sort_unstable_by_key(|&(lx, _)| lx);

        let mut r = rng.f64() * sum;
        for &(lx, w) in &probs { r -= w; if r <= 0.0 { return lx; } }
        probs.last().map(|x| x.0).unwrap_or(0)
    }

    pub fn sample_pos(&mut self, history: &[Token7], clause_depths: &[u8], temp: f32, explore: f32, rng: &mut fastrand::Rng) -> u8 {
        let lex = self.sample_lex(history, clause_depths, temp, explore, rng);
        if lex == 0 && !history.is_empty() {
            // fallback: repeat last class
            return history.last().unwrap().morph_class();
        }
        // Derive class from lex via unigram — but we may not have this mapping.
        // Return 0 (S) and let guess_pos handle it.
        0
    }

    pub fn reinforce(&mut self, tokens: &[Token7], clause_depths: &[u8], delta: f32) {
        for i in 1..tokens.len() {
            let t = &tokens[i];
            let max_n = self.order.min(i);
            for n in 1..=max_n {
                let ctx: Vec<u128> = (i - n..i)
                    .map(|j| pack8d(&tokens[j], clause_depths.get(j).copied().unwrap_or(0)))
                    .collect();
                // `reinforce` escreve. Shards são imutáveis em inferência, então
                // no modo spill aplicamos o delta em `pending` e ele vai para o
                // próximo flush — o shard já gravado não é reescrito.
                let k = chave(&ctx);
                if self.spill_enabled {
                    self.pending.entry(k).or_default().add(t.lex, delta);
                    *self.pending_totals.entry(k).or_insert(0.0) += delta;
                } else if let Some(cands) = self.table.get_mut(&k) {
                    if let Some(c) = cands.get_mut(t.lex) {
                        *c = (*c + delta).max(0.01);
                    }
                }
            }
        }
    }

    // ─── Save / Load ───

    pub fn save(&self, data: &mut Vec<u8>) {
        data.extend_from_slice(&(self.order as u32).to_le_bytes());
        // Salvar lambdas aprendidos por EM
        for &l in &self.lambda { data.extend_from_slice(&l.to_le_bytes()); }
        Self::save_map_lex(&mut *data, &self.table, &self.totals);
        Self::save_unigram(&mut *data, &self.unigram, self.total_lex);
    }

    pub fn load(&mut self, data: &[u8], off: &mut usize) {
        self.order = u32::from_le_bytes(data[*off..*off+4].try_into().unwrap()) as usize;
        *off += 4;
        self.lambda.clear();
        for _ in 0..self.order {
            let l = f32::from_le_bytes(data[*off..*off+4].try_into().unwrap());
            *off += 4;
            self.lambda.push(l);
        }
        Self::load_map_lex(&data, off, &mut self.table, &mut self.totals);
        Self::load_unigram(&data, off, &mut self.unigram, &mut self.total_lex);
    }

    fn save_map_lex(data: &mut Vec<u8>, map: &FxHashMap<CtxKey, Cands>, totals: &FxHashMap<CtxKey, f32>) {
        data.extend_from_slice(&(map.len() as u32).to_le_bytes());
        for (ctx, nexts) in map {
            data.extend_from_slice(&ctx.to_le_bytes());
            data.extend_from_slice(&totals.get(ctx).copied().unwrap_or(0.0).to_le_bytes());
            data.extend_from_slice(&(nexts.len() as u32).to_le_bytes());
            for &(k, v) in &nexts.items {
                data.extend_from_slice(&k.to_le_bytes());
                data.extend_from_slice(&v.to_le_bytes());
            }
        }
    }

    fn load_map_lex(data: &[u8], off: &mut usize, map: &mut FxHashMap<CtxKey, Cands>, totals: &mut FxHashMap<CtxKey, f32>) {
        let n = u32::from_le_bytes(data[*off..*off+4].try_into().unwrap()) as usize;
        *off += 4;
        for _ in 0..n {
            let ctx: CtxKey = u64::from_le_bytes(data[*off..*off+8].try_into().unwrap());
            *off += 8;
            let total = f32::from_le_bytes(data[*off..*off+4].try_into().unwrap());
            *off += 4;
            let n_next = u32::from_le_bytes(data[*off..*off+4].try_into().unwrap()) as usize;
            *off += 4;
            let mut items = Vec::with_capacity(n_next);
            for _ in 0..n_next {
                let k = u32::from_le_bytes(data[*off..*off+4].try_into().unwrap()); *off += 4;
                let v = f32::from_le_bytes(data[*off..*off+4].try_into().unwrap()); *off += 4;
                items.push((k, v));
            }
            items.sort_unstable_by_key(|&(l, _)| l);
            totals.insert(ctx, total);
            map.insert(ctx, Cands { items });
        }
    }

    fn save_unigram(data: &mut Vec<u8>, unigram: &FxHashMap<u32, f32>, total: f32) {
        data.extend_from_slice(&(unigram.len() as u32).to_le_bytes());
        data.extend_from_slice(&total.to_le_bytes());
        for (&k, &v) in unigram { data.extend_from_slice(&k.to_le_bytes()); data.extend_from_slice(&v.to_le_bytes()); }
    }

    fn load_unigram(data: &[u8], off: &mut usize, unigram: &mut FxHashMap<u32, f32>, total: &mut f32) {
        let n = u32::from_le_bytes(data[*off..*off+4].try_into().unwrap()) as usize;
        *off += 4;
        *total = f32::from_le_bytes(data[*off..*off+4].try_into().unwrap());
        *off += 4;
        for _ in 0..n {
            let k = u32::from_le_bytes(data[*off..*off+4].try_into().unwrap()); *off += 4;
            let v = f32::from_le_bytes(data[*off..*off+4].try_into().unwrap()); *off += 4;
            unigram.insert(k, v);
        }
    }
}
