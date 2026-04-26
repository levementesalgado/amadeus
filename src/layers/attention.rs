use crate::layers::rope::apply_rope;
use crate::model::config::ModelConfig;
use crate::model::KvCache;
use crate::tensor::ops::{matmul, matmul_trans_b, softmax_inplace};
use crate::tensor::Tensor;

pub struct Attention {
    pub wq: Tensor,
    pub wk: Tensor,
    pub wv: Tensor,
    pub wo: Tensor,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub rope_theta: f32,
}

impl Attention {
    pub fn new(config: &ModelConfig) -> Self {
        let ne = config.n_embd;
        Self {
            wq: Tensor::zeros(vec![config.n_heads * config.head_dim, ne]),
            wk: Tensor::zeros(vec![config.n_kv_heads * config.head_dim, ne]),
            wv: Tensor::zeros(vec![config.n_kv_heads * config.head_dim, ne]),
            wo: Tensor::zeros(vec![ne, config.n_heads * config.head_dim]),
            n_heads: config.n_heads,
            n_kv_heads: config.n_kv_heads,
            head_dim: config.head_dim,
            rope_theta: config.rope_theta,
        }
    }

    pub fn forward(&self, x: &mut Tensor, cache: &mut KvCache, pos: usize) {
        let hd = self.head_dim;
        let nh = self.n_heads;
        let nkv = self.n_kv_heads;
        let group = nh / nkv;

        let q = matmul(&self.wq, x);
        let k = matmul(&self.wk, x);
        let v = matmul(&self.wv, x);

        let mut q = q.reshape(vec![nh, 1, hd]);
        let mut k = k.reshape(vec![nkv, 1, hd]);
        let v = v.reshape(vec![nkv, 1, hd]); // not mutated later

        apply_rope(&mut q, pos, self.rope_theta);
        apply_rope(&mut k, pos, self.rope_theta);

        let k_flat = k.reshape(vec![nkv * hd]);
        let v_flat = v.reshape(vec![nkv * hd]);

        // Split cache borrows manually
        let kv_stride = nkv * hd;
        let kv_off = pos * kv_stride;
        let seq_len = (pos + 1).max(cache.seq_len);

        // Write into cache
        let k_cache = &mut cache.keys;
        let v_cache = &mut cache.values;
        k_cache[kv_off..kv_off + kv_stride].copy_from_slice(k_flat.as_slice());
        v_cache[kv_off..kv_off + kv_stride].copy_from_slice(v_flat.as_slice());
        cache.seq_len = seq_len;

        // Views into cache (already stored in k_cache, v_cache)

        // GQA
        let mut attn_out = Tensor::zeros(vec![nh, hd]);
        let scale = 1.0 / (hd as f32).sqrt();

        for h in 0..nh {
            let kv_h = h / group;
            let q_h = &q.as_slice()[h * hd..(h + 1) * hd];
            let q_t = Tensor::from_vec(q_h.to_vec(), vec![1, hd]);

            let mut k_s = Vec::with_capacity(seq_len * hd);
            for s in 0..seq_len {
                let base = s * kv_stride + kv_h * hd;
                k_s.extend_from_slice(&k_cache[base..base + hd]);
            }
            let k_s = Tensor::from_vec(k_s, vec![seq_len, hd]);

            let mut scores = matmul_trans_b(&q_t, &k_s);
            for s in 0..seq_len {
                scores.as_mut_slice()[s] *= scale;
                if s > pos {
                    scores.as_mut_slice()[s] = f32::NEG_INFINITY;
                }
            }
            softmax_inplace(&mut scores);

            let mut v_s = Vec::with_capacity(seq_len * hd);
            for s in 0..seq_len {
                let base = s * kv_stride + kv_h * hd;
                v_s.extend_from_slice(&v_cache[base..base + hd]);
            }
            let v_s = Tensor::from_vec(v_s, vec![seq_len, hd]);

            let out_h = matmul(&scores, &v_s);
            let start = h * hd;
            attn_out.as_mut_slice()[start..start + hd]
                .copy_from_slice(out_h.as_slice());
        }

        let attn_flat = attn_out.reshape(vec![nh * hd, 1]);
        let result = matmul(&self.wo, &attn_flat);
        x.as_mut_slice().copy_from_slice(result.as_slice());
    }
}
