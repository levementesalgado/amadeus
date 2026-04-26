use crate::model::config::ModelConfig;
use crate::tensor::ops::{matmul, silu_inplace, mul_inplace};
use crate::tensor::Tensor;

/// SwiGLU feed-forward network:
///   out = down(silu(gate @ x) * (up @ x))
pub struct Ffn {
    pub gate: Tensor,
    pub up: Tensor,
    pub down: Tensor,
}

impl Ffn {
    pub fn new(config: &ModelConfig) -> Self {
        let n_embd = config.n_embd;
        let n_int = config.n_intermediate;
        Self {
            gate: Tensor::zeros(vec![n_int, n_embd]),
            up: Tensor::zeros(vec![n_int, n_embd]),
            down: Tensor::zeros(vec![n_embd, n_int]),
        }
    }

    /// x: [1, n_embd], modified in-place.
    pub fn forward(&self, x: &mut Tensor) {
        // gate @ x
        let mut gate_out = matmul(&self.gate, x); // [1, n_int]
        silu_inplace(&mut gate_out);

        // up @ x
        let up_out = matmul(&self.up, x); // [1, n_int]

        // element-wise multiply
        mul_inplace(&mut gate_out, &up_out); // [1, n_int]

        // down @ result
        let down_out = matmul(&self.down, &gate_out); // [1, n_embd]

        x.as_mut_slice().copy_from_slice(down_out.as_slice());
    }
}
