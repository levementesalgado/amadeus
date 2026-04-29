/// Hako compiler integration.
///
/// The long-term goal: write a compiler that translates Amadeus
/// inference kernels into Hako (Mizu OS's native language), so
/// that LLM inference runs as a first-class citizen on the kernel.
///
/// This module will grow as the Hako compiler takes shape.

/// Hako IR node types (planned).
pub enum HakoIrNode {
    /// Load a model weight from a named tensor.
    LoadTensor(String),
    /// Matrix-vector multiply (GEMV).
    Gemv {
        input_reg: usize,
        weight_tensor: String,
        output_reg: usize,
        quant: QuantMode,
    },
    /// Element-wise operation.
    ElementWise {
        op: ElemOp,
        input_reg: usize,
        output_reg: usize,
    },
    /// Call into a native Amadeus kernel (fallback).
    NativeCall(String),
}

/// Quantization mode for Hako GEMV instructions.
pub enum QuantMode {
    F32,
    F16,
    Q4_0,
}

/// Element-wise operations.
pub enum ElemOp {
    Relu,
    Silu,
    Softmax,
    RmsNorm,
}

/// A Hako compilation unit.
pub struct HakoModule {
    pub name: String,
    pub nodes: Vec<HakoIrNode>,
}

impl HakoModule {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.into(),
            nodes: Vec::new(),
        }
    }

    /// Lower an Amadeus attention layer to Hako IR.
    pub fn lower_attention(&mut self) {
        todo!("Lower attention to Hako IR")
    }

    /// Lower an Amadeus FFN layer to Hako IR.
    pub fn lower_ffn(&mut self) {
        todo!("Lower FFN to Hako IR")
    }

    /// Emit Hako assembly from the IR.
    pub fn emit(&self) -> String {
        todo!("Hako assembly emission")
    }
}
