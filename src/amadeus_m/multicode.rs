use crate::tensor::Tensor;
use super::rng;

/// Três camadas de codificação por token, combinadas por gating dinâmico.
/// A cada forward, os pesos do gate são modulados pelo estado de excursão.
pub struct MultiCoding {
    /// Embedding sintático: estrutura gramatical, posição, função
    pub syntax: Tensor,
    /// Embedding semântico: significado conceitual (mesma dimensão do modelo)
    pub semantic: Tensor,
    /// Embedding afetivo: valência, ativação, carga emocional
    pub affect_val: Tensor,
    /// Gate: projeta [excursion_phase, curiosity, focus_attention] → 3 pesos
    pub gate_proj: Tensor,
    /// Dimensão dos embeddings
    pub n_embd: usize,
    /// Dimensão do espaço afetivo (pode ser menor)
    pub d_affect: usize,
}

impl MultiCoding {
    pub fn new(vocab_size: usize, n_embd: usize, d_affect: usize, rng: &mut fastrand::Rng) -> Self {
        Self {
            syntax: rng::init_embedding(vocab_size, n_embd, rng),
            semantic: rng::init_embedding(vocab_size, n_embd, rng),
            affect_val: rng::init_embedding(vocab_size, d_affect, rng),
            gate_proj: rng::init_linear(3, 3, rng), // [phase, curiosity, focus] → [w_syn, w_sem, w_aff]
            n_embd,
            d_affect,
        }
    }

    /// Produz o embedding combinado para um token, dado o estado de consciência
    pub fn forward(
        &self,
        token: usize,
        excursion_phase: f32,
        curiosity: f32,
        focus_attention: f32,
    ) -> Vec<f32> {
        let g = self.compute_gate(excursion_phase, curiosity, focus_attention);
        let w_syn = g[0];
        let w_sem = g[1];
        let w_aff = g[2];

        let syn = self.embed(&self.syntax, token, self.n_embd);
        let sem = self.embed(&self.semantic, token, self.n_embd);
        let aff = self.embed(&self.affect_val, token, self.d_affect);

        // Soma ponderada: sintático + semântico + afetivo (broadcast do afetivo)
        let mut out = Vec::with_capacity(self.n_embd);
        for i in 0..self.n_embd {
            let a = aff[i % self.d_affect];
            out.push(w_syn * syn[i] + w_sem * sem[i] + w_aff * a);
        }
        out
    }

    /// Calcula os pesos do gate: [w_syntax, w_semantic, w_affect]
    /// Excursão alta → mais peso no afetivo
    /// Foco alto → mais peso no semântico
    /// Surpresa alta → mais peso no sintático
    pub fn compute_gate(&self, phase: f32, curiosity: f32, focus: f32) -> [f32; 3] {
        let input = Tensor::from_vec(vec![phase, curiosity, focus], vec![1, 3]);
        let raw = crate::tensor::ops::matmul(&self.gate_proj, &input);
        let slice = raw.as_slice();
        softmax3(slice[0], slice[1], slice[2])
    }

    fn embed<'a>(&self, table: &'a Tensor, token: usize, dim: usize) -> &'a [f32] {
        let start = token * dim;
        let end = (start + dim).min(table.as_slice().len());
        &table.as_slice()[start..end]
    }
}

fn softmax3(a: f32, b: f32, c: f32) -> [f32; 3] {
    let max = a.max(b).max(c);
    let ea = (a - max).exp();
    let eb = (b - max).exp();
    let ec = (c - max).exp();
    let sum = ea + eb + ec;
    if sum > 0.0 {
        [ea / sum, eb / sum, ec / sum]
    } else {
        [0.5, 0.3, 0.2]
    }
}
