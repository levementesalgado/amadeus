use crate::tensor::Tensor;

/// C = A @ B   where A: [M, K], B: [K, N]
/// If B is [1, K] (row vector), it is treated as [K, 1] (column).
/// Parallelized with rayon over the M dimension.
pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor {
    use rayon::prelude::*;

    let a_shape = a.shape();
    let b_shape = b.shape();
    assert_eq!(a_shape.len(), 2, "matmul A must be 2D");
    assert_eq!(b_shape.len(), 2, "matmul B must be 2D");

    let m = a_shape[0];
    let k = a_shape[1];

    let (n, b_is_row) = if b_shape[0] == 1 && b_shape[1] == k {
        (1, true)
    } else {
        assert_eq!(k, b_shape[0], "matmul inner dim mismatch: {} vs {}", k, b_shape[0]);
        (b_shape[1], false)
    };

    let av = a.view();
    let bv = b.view();
    let mut c = Tensor::zeros(vec![m, n]);
    let cs: &mut [f32] = c.as_mut_slice();

    cs.par_chunks_mut(n).enumerate().for_each(|(i, row_out)| {
        let a_row = av.row(i);
        for j in 0..n {
            let mut sum = 0.0;
            for t in 0..k {
                let b_val = if b_is_row {
                    bv.data[t]
                } else {
                    bv.row(t)[j]
                };
                sum += a_row[t] * b_val;
            }
            row_out[j] = sum;
        }
    });
    c
}

/// C = A @ B^T   where A: [M, K], B: [N, K]
pub fn matmul_trans_b(a: &Tensor, b: &Tensor) -> Tensor {
    use rayon::prelude::*;

    let a_shape = a.shape();
    let b_shape = b.shape();
    assert_eq!(a_shape.len(), 2);
    assert_eq!(b_shape.len(), 2);
    assert_eq!(a_shape[1], b_shape[1], "matmul_trans_b inner dim mismatch");

    let m = a_shape[0];
    let k = a_shape[1];
    let n = b_shape[0];

    let av = a.view();
    let bv = b.view();
    let mut c = Tensor::zeros(vec![m, n]);
    let cs: &mut [f32] = c.as_mut_slice();

    cs.par_chunks_mut(n).enumerate().for_each(|(i, row_out)| {
        let a_row = av.row(i);
        for j in 0..n {
            let b_row = bv.row(j);
            let mut sum = 0.0;
            for t in 0..k {
                sum += a_row[t] * b_row[t];
            }
            row_out[j] = sum;
        }
    });
    c
}

/// In-place add: self += other (broadcast over last dim).
pub fn add_inplace(a: &mut Tensor, b: &Tensor) {
    assert_eq!(a.numel(), b.numel(), "add_inplace: size mismatch");
    for (x, y) in a.as_mut_slice().iter_mut().zip(b.as_slice()) {
        *x += y;
    }
}

/// Element-wise silu (SiLU/Swish): f(x) = x * sigmoid(x)
pub fn silu_inplace(t: &mut Tensor) {
    for x in t.as_mut_slice() {
        *x = *x / (1.0 + (-*x).exp());
    }
}

/// Element-wise mul: a *= b (broadcast over last dim)
pub fn mul_inplace(a: &mut Tensor, b: &Tensor) {
    assert_eq!(a.numel(), b.numel());
    for (x, y) in a.as_mut_slice().iter_mut().zip(b.as_slice()) {
        *x *= y;
    }
}

/// Softmax over last dimension (in-place).
/// Safely handles all-NEG_INFINITY rows (returns uniform).
pub fn softmax_inplace(t: &mut Tensor) {
    let shape = t.shape().to_vec();
    let last = shape.last().copied().unwrap_or(1);
    let outer: usize = shape.iter().rev().skip(1).product();
    let slice = t.as_mut_slice();

    for i in 0..outer {
        let start = i * last;
        let row = &mut slice[start..start + last];
        let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        if max_val == f32::NEG_INFINITY {
            for x in row.iter_mut() { *x = 1.0 / last as f32; }
            continue;
        }
        let sum: f32 = row.iter().map(|x| (x - max_val).exp()).sum();
        let inv = 1.0 / sum;
        for x in row.iter_mut() {
            *x = (*x - max_val).exp() * inv;
        }
    }
}

/// RMS norm: output = x * rsqrt(mean(x^2) + eps)  (in-place on last dim)
pub fn rms_norm_inplace(t: &mut Tensor, weight: &[f32], eps: f32) {
    let last = t.last_dim();
    let outer = t.numel() / last;
    let data = t.as_mut_slice();

    for i in 0..outer {
        let start = i * last;
        let row = &mut data[start..start + last];
        let sum_sq: f32 = row.iter().map(|v| v * v).sum();
        let scale = 1.0 / ((sum_sq / last as f32) + eps).sqrt();
        for j in 0..last {
            row[j] *= scale * weight[j];
        }
    }
}

/// Argmax over last dimension, returns index into the *first* row.
/// Treats NaN as -inf so it never wins.
pub fn argmax(slice: &[f32], last_dim: usize) -> usize {
    slice[0..last_dim]
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            let a = if a.is_nan() { &f32::NEG_INFINITY } else { a };
            let b = if b.is_nan() { &f32::NEG_INFINITY } else { b };
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(i, _)| i)
        .unwrap_or(0)
}
