use crate::tensor::Tensor;

/// Apply rotary position embeddings (RoPE) to query or key tensor.
///
/// x: [n_heads, seq_len, head_dim]  (or [n_kv_heads, seq_len, head_dim])
/// pos: absolute position of the first token in the sequence.
///
/// Modifies x in-place by rotating pairs of dimensions.
pub fn apply_rope(x: &mut Tensor, pos: usize, theta: f32) {
    let shape = x.shape().to_vec();
    assert!(shape.len() >= 2, "RoPE needs at least 2D");
    let head_dim = shape[shape.len() - 1];
    let seq_len = shape[shape.len() - 2];
    let outer: usize = shape[..shape.len() - 2].iter().product();

    let half = head_dim / 2;
    let data = x.as_mut_slice();
    let row_len = seq_len * head_dim;
    let inner_len = head_dim;

    for batch in 0..outer {
        let base = batch * row_len;
        for s in 0..seq_len {
            let p = pos + s;
            let row_start = base + s * inner_len;
            for i in 0..half {
                let a_idx = row_start + i;
                let b_idx = row_start + i + half;
                let freq = p as f32 / (theta.powf(2.0 * i as f32 / head_dim as f32));
                let (sin, cos) = freq.sin_cos();
                let a = data[a_idx];
                let b = data[b_idx];
                data[a_idx] = a * cos - b * sin;
                data[b_idx] = a * sin + b * cos;
            }
        }
    }
}
