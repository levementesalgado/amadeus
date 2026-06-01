use crate::quant::QuantType;
use byteorder::{LittleEndian, ReadBytesExt};
use std::io::{Cursor, Read};

// "GGUF" as little-endian u32: bytes G G U F → 0x46554747
pub const GGUF_MAGIC: u32 = 0x46554747;

#[derive(Debug, Clone, PartialEq)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
    U64(u64),
    I64(i64),
    F64(f64),
}

#[derive(Debug, Clone)]
pub struct GgufKv {
    pub key: String,
    pub value: GgufValue,
}

#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    pub name: String,
    pub n_dims: u32,
    pub shape: Vec<u64>,
    pub quant_type: QuantType,
    pub offset: u64,
}

#[derive(Debug)]
pub struct GgufHeader {
    pub version: u32,
    pub n_tensors: u64,
    pub n_kv: u64,
    pub kv_pairs: Vec<GgufKv>,
    pub tensor_infos: Vec<GgufTensorInfo>,
    pub data_offset: u64,
}

fn read_string(cursor: &mut Cursor<&[u8]>) -> Result<String, String> {
    let len = cursor.read_u64::<LittleEndian>().map_err(|e| format!("str len: {e}"))? as usize;
    let mut buf = vec![0u8; len];
    cursor.read_exact(&mut buf).map_err(|e| format!("str data: {e}"))?;
    String::from_utf8(buf).map_err(|e| format!("utf8: {e}"))
}

fn read_value(cursor: &mut Cursor<&[u8]>) -> Result<GgufValue, String> {
    let ty = cursor.read_u32::<LittleEndian>().map_err(|e| format!("type: {e}"))?;
    read_typed_value(cursor, ty)
}

fn read_typed_value(cursor: &mut Cursor<&[u8]>, ty: u32) -> Result<GgufValue, String> {
    match ty {
        0 => Ok(GgufValue::U8(cursor.read_u8().map_err(|e| format!("u8: {e}"))?)),
        1 => Ok(GgufValue::I8(cursor.read_i8().map_err(|e| format!("i8: {e}"))?)),
        2 => Ok(GgufValue::U16(cursor.read_u16::<LittleEndian>().map_err(|e| format!("u16: {e}"))?)),
        3 => Ok(GgufValue::I16(cursor.read_i16::<LittleEndian>().map_err(|e| format!("i16: {e}"))?)),
        4 => Ok(GgufValue::U32(cursor.read_u32::<LittleEndian>().map_err(|e| format!("u32: {e}"))?)),
        5 => Ok(GgufValue::I32(cursor.read_i32::<LittleEndian>().map_err(|e| format!("i32: {e}"))?)),
        6 => Ok(GgufValue::F32(cursor.read_f32::<LittleEndian>().map_err(|e| format!("f32: {e}"))?)),
        7 => Ok(GgufValue::Bool(cursor.read_u8().map_err(|e| format!("bool: {e}"))? != 0)),
        8 => Ok(GgufValue::String(read_string(cursor)?)),
        9 => {
            let arr_ty = cursor.read_u32::<LittleEndian>().map_err(|e| format!("arr type: {e}"))?;
            let arr_len = cursor.read_u64::<LittleEndian>().map_err(|e| format!("arr len: {e}"))? as usize;
            let mut items = Vec::with_capacity(arr_len);
            for _ in 0..arr_len {
                items.push(read_typed_value(cursor, arr_ty)?);
            }
            Ok(GgufValue::Array(items))
        }
        10 => Ok(GgufValue::U64(cursor.read_u64::<LittleEndian>().map_err(|e| format!("u64: {e}"))?)),
        11 => Ok(GgufValue::I64(cursor.read_i64::<LittleEndian>().map_err(|e| format!("i64: {e}"))?)),
        12 => Ok(GgufValue::F64(cursor.read_f64::<LittleEndian>().map_err(|e| format!("f64: {e}"))?)),
        _ => Err(format!("unknown GGUF value type: {ty}")),
    }
}

fn quant_type_from_ggml(raw: u32) -> Result<QuantType, String> {
    match raw {
        0 => Ok(QuantType::F32),
        1 => Ok(QuantType::F16),
        2 => Ok(QuantType::Q4_0),
        8 => Ok(QuantType::Q8_0),
        14 => Ok(QuantType::Q6_K),
        other => Err(format!("unsupported GGML quant type: {other}")),
    }
}

fn read_tensor_info(cursor: &mut Cursor<&[u8]>) -> Result<GgufTensorInfo, String> {
    let name = read_string(cursor)?;
    let n_dims = cursor.read_u32::<LittleEndian>().map_err(|e| format!("n_dims: {e}"))?;
    let mut shape = Vec::with_capacity(n_dims as usize);
    for _ in 0..n_dims {
        shape.push(cursor.read_u64::<LittleEndian>().map_err(|e| format!("shape: {e}"))?);
    }
    let quant_raw = cursor.read_u32::<LittleEndian>().map_err(|e| format!("quant: {e}"))?;
    let quant_type = quant_type_from_ggml(quant_raw)?;
    let offset = cursor.read_u64::<LittleEndian>().map_err(|e| format!("offset: {e}"))?;
    Ok(GgufTensorInfo { name, n_dims, shape, quant_type, offset })
}

// ═══════════════════════════════════════════════════════
// GGUF Writer
// ═══════════════════════════════════════════════════════

pub struct GgufWriter {
    buf: Vec<u8>,
}

impl GgufWriter {
    pub fn new() -> Self {
        Self { buf: Vec::with_capacity(1 << 20) }
    }

    pub fn finish(self) -> Vec<u8> { self.buf }

    fn write_u8(&mut self, v: u8) { self.buf.push(v); }
    fn write_u16(&mut self, v: u16) { self.buf.extend_from_slice(&v.to_le_bytes()); }
    fn write_u32(&mut self, v: u32) { self.buf.extend_from_slice(&v.to_le_bytes()); }
    fn write_u64(&mut self, v: u64) { self.buf.extend_from_slice(&v.to_le_bytes()); }
    fn write_f32(&mut self, v: f32) { self.buf.extend_from_slice(&v.to_le_bytes()); }

    fn write_string(&mut self, s: &str) {
        self.write_u64(s.len() as u64);
        self.buf.extend_from_slice(s.as_bytes());
    }

    fn write_kv(&mut self, key: &str, value: &GgufValue) {
        self.write_string(key);
        self.write_value(value);
    }

    fn write_value(&mut self, value: &GgufValue) {
        match value {
            GgufValue::U8(v) => { self.write_u32(0); self.write_u8(*v); }
            GgufValue::I8(v) => { self.write_u32(1); self.buf.push(*v as u8); }
            GgufValue::U16(v) => { self.write_u32(2); self.write_u16(*v); }
            GgufValue::I16(v) => { self.write_u32(3); self.buf.extend_from_slice(&v.to_le_bytes()); }
            GgufValue::U32(v) => { self.write_u32(4); self.write_u32(*v); }
            GgufValue::I32(v) => { self.write_u32(5); self.buf.extend_from_slice(&v.to_le_bytes()); }
            GgufValue::F32(v) => { self.write_u32(6); self.write_f32(*v); }
            GgufValue::Bool(v) => { self.write_u32(7); self.write_u8(if *v { 1 } else { 0 }); }
            GgufValue::String(s) => {
                self.write_u32(8);
                self.write_string(s);
            }
            GgufValue::Array(items) => {
                self.write_u32(9);
                // Determine element type from first item
                let elem_type = if items.is_empty() { 4u32 } else {
                    match &items[0] {
                        GgufValue::U8(_) => 0,
                        GgufValue::I8(_) => 1,
                        GgufValue::U16(_) => 2,
                        GgufValue::I16(_) => 3,
                        GgufValue::U32(_) => 4,
                        GgufValue::I32(_) => 5,
                        GgufValue::F32(_) => 6,
                        GgufValue::Bool(_) => 7,
                        GgufValue::String(_) => 8,
                        GgufValue::Array(_) => 9,
                        GgufValue::U64(_) => 10,
                        GgufValue::I64(_) => 11,
                        GgufValue::F64(_) => 12,
                    }
                };
                self.write_u32(elem_type);
                self.write_u64(items.len() as u64);
                for item in items { self.write_value(item); }
            }
            GgufValue::U64(v) => { self.write_u32(10); self.write_u64(*v); }
            GgufValue::I64(v) => { self.write_u32(11); self.buf.extend_from_slice(&v.to_le_bytes()); }
            GgufValue::F64(v) => { self.write_u32(12); self.buf.extend_from_slice(&v.to_le_bytes()); }
        }
    }

    pub fn write_header(&mut self, version: u32, n_tensors: u64, n_kv: u64, kv: &[GgufKv], tensors: &[GgufTensorInfo]) {
        self.write_u32(GGUF_MAGIC);
        self.write_u32(version);
        self.write_u64(n_tensors);
        self.write_u64(n_kv);
        for k in kv { self.write_kv(&k.key, &k.value); }
        for t in tensors {
            self.write_string(&t.name);
            self.write_u32(t.n_dims);
            for &d in &t.shape { self.write_u64(d); }
            self.write_u32(quant_type_to_ggml(t.quant_type));
            self.write_u64(t.offset);
        }
    }

    pub fn write_tensor_data(&mut self, data: &[u8]) {
        // Pad to 32-byte alignment
        let align = 32;
        let rem = self.buf.len() % align;
        if rem != 0 {
            self.buf.extend(std::iter::repeat(0u8).take(align - rem));
        }
        self.buf.extend_from_slice(data);
    }

    pub fn write_raw(&mut self, data: &[u8]) {
        self.buf.extend_from_slice(data);
    }

    pub fn current_offset(&self) -> u64 { self.buf.len() as u64 }
}

fn quant_type_to_ggml(qt: QuantType) -> u32 {
    match qt {
        QuantType::F32 => 0,
        QuantType::F16 => 1,
        QuantType::Q4_0 => 2,
        QuantType::Q8_0 => 8,
        QuantType::Q6_K => 14,
    }
}

pub fn parse_header(data: &[u8]) -> Result<GgufHeader, String> {
    let mut cursor = Cursor::new(data);

    let magic = cursor.read_u32::<LittleEndian>().map_err(|e| format!("magic: {e}"))?;
    if magic != GGUF_MAGIC {
        return Err(format!("bad magic: 0x{magic:08X}"));
    }

    let version = cursor.read_u32::<LittleEndian>().map_err(|e| format!("version: {e}"))?;

    let (n_tensors, n_kv) = if version == 1 {
        let nt = cursor.read_u32::<LittleEndian>().map_err(|e| format!("n_tensors(v1): {e}"))? as u64;
        let nk = cursor.read_u32::<LittleEndian>().map_err(|e| format!("n_kv(v1): {e}"))? as u64;
        (nt, nk)
    } else {
        let nt = cursor.read_u64::<LittleEndian>().map_err(|e| format!("n_tensors: {e}"))?;
        let nk = cursor.read_u64::<LittleEndian>().map_err(|e| format!("n_kv: {e}"))?;
        (nt, nk)
    };

    let mut kv_pairs = Vec::with_capacity(n_kv as usize);
    for _ in 0..n_kv {
        let key = read_string(&mut cursor)?;
        let value = read_value(&mut cursor)?;
        kv_pairs.push(GgufKv { key, value });
    }

    let mut tensor_infos = Vec::with_capacity(n_tensors as usize);
    for _ in 0..n_tensors {
        tensor_infos.push(read_tensor_info(&mut cursor)?);
    }

    let data_offset = cursor.position();

    Ok(GgufHeader { version, n_tensors, n_kv, kv_pairs, tensor_infos, data_offset })
}
