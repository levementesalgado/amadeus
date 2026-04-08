use crate::tensor::ops::{matmul, matmul_trans_b, softmax_inplace};
use crate::tensor::Tensor;
use super::config::AmadeusMConfig;

pub struct HabitMemory {
    pub wq: Tensor,
    pub wk: Tensor,
    pub wv: Tensor,
    pub wo: Tensor,
    pub norm_weight: Tensor,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub window_size: usize,
    pub cache_k: Vec<f32>,
    pub cache_v: Vec<f32>,
    pub seq_len: usize,
}

impl HabitMemory {
    pub fn new(cfg: &AmadeusMConfig, rng: &mut fastrand::Rng) -> Self {
        let n_embd = cfg.n_embd;
        let n_heads = cfg.n_heads;
        let n_kv = cfg.n_kv_heads;
        let hd = cfg.head_dim;
        Self {
            wq: super::rng::init_linear(n_embd, n_heads * hd, rng),
            wk: super::rng::init_linear(n_embd, n_kv * hd, rng),
            wv: super::rng::init_linear(n_embd, n_kv * hd, rng),
            wo: super::rng::init_linear(n_heads * hd, n_embd, rng),
            norm_weight: super::rng::init_rms_weight(n_embd),
            n_heads,
            n_kv_heads: n_kv,
            head_dim: hd,
            window_size: cfg.window_size,
            cache_k: vec![0.0; cfg.max_seq_len * n_kv * hd],
            cache_v: vec![0.0; cfg.max_seq_len * n_kv * hd],
            seq_len: 0,
        }
    }

    pub fn attend(&mut self, x: &[f32], pos: usize, n_embd: usize) -> Vec<f32> {
        let hd = self.head_dim;
        let nh = self.n_heads;
        let nkv = self.n_kv_heads;
        let group = nh / nkv;

        let x_t = Tensor::from_vec(x.to_vec(), vec![1, n_embd]);

        let q = matmul(&self.wq, &x_t);
        let k = matmul(&self.wk, &x_t);
        let v = matmul(&self.wv, &x_t);

        let kv_stride = nkv * hd;
        let kv_off = pos * kv_stride;
        let k_data = k.as_slice();
        let v_data = v.as_slice();

        if pos < self.cache_k.len() / kv_stride {
            let end = (kv_off + kv_stride).min(self.cache_k.len());
            self.cache_k[kv_off..end].copy_from_slice(&k_data[..kv_stride.min(end - kv_off)]);
            self.cache_v[kv_off..end].copy_from_slice(&v_data[..kv_stride.min(end - kv_off)]);
        }
        self.seq_len = self.seq_len.max(pos + 1);

        let max_cached = self.cache_k.len() / kv_stride;
        let effective_seq = self.seq_len.min(max_cached);
        let start_pos = if effective_seq > self.window_size { effective_seq - self.window_size } else { 0 };
        let eff_len = effective_seq - start_pos;

        let mut output = vec![0.0; nh * hd];

        for h in 0..nh {
            let kv_h = h / group;
            let q_start = h * hd;
            let q_row: Vec<f32> = q.as_slice()[q_start..q_start + hd].to_vec();
            let q_t = Tensor::from_vec(q_row, vec![1, hd]);

            let mut k_s = Vec::with_capacity(eff_len * hd);
            let mut v_s = Vec::with_capacity(eff_len * hd);
            for s in start_pos..effective_seq {
                let base = s * kv_stride + kv_h * hd;
                k_s.extend_from_slice(&self.cache_k[base..base + hd]);
                v_s.extend_from_slice(&self.cache_v[base..base + hd]);
            }
            let k_t = Tensor::from_vec(k_s, vec![eff_len, hd]);
            let v_t = Tensor::from_vec(v_s, vec![eff_len, hd]);

            let mut scores = matmul_trans_b(&q_t, &k_t);
            let scale = 1.0 / (hd as f32).sqrt();
            for s in 0..eff_len {
                scores.as_mut_slice()[s] *= scale;
                if start_pos + s > pos {
                    scores.as_mut_slice()[s] = f32::NEG_INFINITY;
                }
            }
            softmax_inplace(&mut scores);
            let out_h = matmul(&scores, &v_t);
            let out_start = h * hd;
            output[out_start..out_start + hd].copy_from_slice(out_h.as_slice());
        }

        let out_t = Tensor::from_vec(output, vec![1, nh * hd]);
        let result = matmul(&self.wo, &out_t);
        result.as_slice().to_vec()
    }

    pub fn reset_cache(&mut self) {
        self.cache_k.fill(0.0);
        self.cache_v.fill(0.0);
        self.seq_len = 0;
    }
}
