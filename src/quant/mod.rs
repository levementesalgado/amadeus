pub mod gguf;

use bytemuck::{Pod, Zeroable};

const QK_K: usize = 256;

/// Supported quantization types.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QuantType {
    F32,
    F16,
    Q4_0,
    Q8_0,
    Q6_K,
}

impl QuantType {
    pub fn block_size(&self) -> usize {
        match self {
            QuantType::F32 => 1,
            QuantType::F16 => 1,
            QuantType::Q4_0 => 32,
            QuantType::Q8_0 => 32,
            QuantType::Q6_K => QK_K,
        }
    }

    pub fn bytes_per_block(&self) -> usize {
        match self {
            QuantType::F32 => 4,
            QuantType::F16 => 2,
            QuantType::Q4_0 => 18,
            QuantType::Q8_0 => 34,
            QuantType::Q6_K => QK_K / 2 + QK_K / 4 + QK_K / 16 + 2,
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "F32" => Some(QuantType::F32),
            "F16" => Some(QuantType::F16),
            "Q4_0" => Some(QuantType::Q4_0),
            "Q8_0" => Some(QuantType::Q8_0),
            "Q6_K" => Some(QuantType::Q6_K),
            _ => None,
        }
    }
}

/// Q4_0 block: 2-byte f16 scale + 16 bytes of packed 4-bit values = 32 weights.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BlockQ4_0 {
    pub scale: u16, // f16
    pub nibbles: [u8; 16],
}

/// Q8_0 block: 2-byte f16 scale + 32 x i8 values.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BlockQ8_0 {
    pub scale: u16, // f16
    pub values: [i8; 32],
}

/// Q6_K block: 256 weights in 210 bytes.
/// ql[128] = lower 4 bits, qh[64] = upper 2 bits, scales[16] = int8 scales, d = f16 super-scale.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BlockQ6_K {
    pub ql: [u8; QK_K / 2],
    pub qh: [u8; QK_K / 4],
    pub scales: [i8; QK_K / 16],
    pub d: u16,
}

fn f16_to_f32(v: u16) -> f32 {
    half::f16::from_bits(v).to_f32()
}

/// Dequantize a slice of quantized bytes into f32 values.
/// `data` must point to the start of the tensor's weight data.
pub fn dequantize(data: &[u8], quant: QuantType, numel: usize) -> Vec<f32> {
    match quant {
        QuantType::F32 => {
            data.chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect()
        }
        QuantType::F16 => {
            data.chunks_exact(2)
                .map(|c| f16_to_f32(u16::from_le_bytes(c.try_into().unwrap())))
                .collect()
        }
        QuantType::Q4_0 => {
            let mut out = Vec::with_capacity(numel);
            for chunk in data.chunks_exact(18) {
                let block: BlockQ4_0 = bytemuck::pod_read_unaligned(chunk);
                let scale = f16_to_f32(block.scale);
                for &nibble in &block.nibbles {
                    let lo = (nibble & 0x0F) as i8;
                    let hi = (nibble >> 4) as i8;
                    out.push((lo as f32 - 8.0) * scale);
                    out.push((hi as f32 - 8.0) * scale);
                }
            }
            out
        }
        QuantType::Q8_0 => {
            let mut out = Vec::with_capacity(numel);
            for chunk in data.chunks_exact(34) {
                let block: BlockQ8_0 = bytemuck::pod_read_unaligned(chunk);
                let scale = f16_to_f32(block.scale);
                for &v in &block.values {
                    out.push((v as f32) * scale);
                }
            }
            out
        }
        QuantType::Q6_K => {
            let bp = quant.bytes_per_block();
            let num_blocks = (numel + QK_K - 1) / QK_K;
            let mut out = vec![0.0f32; numel];
            for bi in 0..num_blocks {
                let off = bi * bp;
                let block: BlockQ6_K = bytemuck::pod_read_unaligned(&data[off..off + bp]);
                let d = f16_to_f32(block.d);
                for half in 0..2 {
                    let bo = bi * QK_K + half * (QK_K / 2);
                    let ql_off = half * (QK_K / 4);
                    let qh_off = half * (QK_K / 8);
                    let sc_off = half * (QK_K / 32);
                    for l in 0..32 {
                        let is = l / 16;
                        let ql_low = block.ql[ql_off + l] as u16;
                        let ql_high = block.ql[ql_off + l + 32] as u16;
                        let qh_b = block.qh[qh_off + l] as u16;

                        let q1 = ((ql_low & 0xF) | ((qh_b >> 0) & 3) << 4) as i8 - 32;
                        let q2 = ((ql_high & 0xF) | ((qh_b >> 2) & 3) << 4) as i8 - 32;
                        let q3 = ((ql_low >> 4) | ((qh_b >> 4) & 3) << 4) as i8 - 32;
                        let q4 = ((ql_high >> 4) | ((qh_b >> 6) & 3) << 4) as i8 - 32;

                        let s0 = block.scales[sc_off + is] as f32;
                        let s2 = block.scales[sc_off + is + 2] as f32;
                        let s4 = block.scales[sc_off + is + 4] as f32;
                        let s6 = block.scales[sc_off + is + 6] as f32;

                        out[bo + l] = d * s0 * q1 as f32;
                        out[bo + l + 32] = d * s2 * q2 as f32;
                        out[bo + l + 64] = d * s4 * q3 as f32;
                        out[bo + l + 96] = d * s6 * q4 as f32;
                    }
                }
            }
            out
        }
    }
}
