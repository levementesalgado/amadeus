use crate::tensor::Tensor;
use fastrand::Rng;

pub fn init_tensor(shape: Vec<usize>, scale: f32, rng: &mut Rng) -> Tensor {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n).map(|_| rng.f32() * scale - scale / 2.0).collect();
    Tensor::from_vec(data, shape)
}

pub fn init_linear(n_in: usize, n_out: usize, rng: &mut Rng) -> Tensor {
    init_tensor(vec![n_out, n_in], (2.0 / n_in as f32).sqrt(), rng)
}

pub fn init_bias(n: usize, _rng: &mut Rng) -> Tensor {
    Tensor::from_vec(vec![0.0; n], vec![n])
}

pub fn init_embedding(vocab: usize, dim: usize, rng: &mut Rng) -> Tensor {
    init_tensor(vec![vocab, dim], 0.02, rng)
}

pub fn init_rms_weight(n: usize) -> Tensor {
    Tensor::from_vec(vec![1.0; n], vec![n])
}
