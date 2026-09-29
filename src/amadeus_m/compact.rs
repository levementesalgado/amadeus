//! Tipos compactos para a tabela de contextos do CUBO.
//!
//! A tabela original usava `HashMap<Vec<u128>, HashMap<u32, f32>>`, que medido
//! custa **266 bytes por entrada** (em 1M de entradas, glibc malloc). Só para
//! 7,67 milhões de contextos são 2,04 GB ao carregar.
//!
//! Onde o custo estava:
//!
//! - **Chave `Vec<u128>`**: 24 bytes inline (ptr, cap, len) mais uma alocação
//!   de heap por entrada. Um `u64` canônico é 8 bytes e zero alocações.
//! - **Valor `HashMap<u32, f32>`**: 48 bytes inline (hashbrown vazio já custa
//!   isso) mais outra alocação. Um `Vec<(u32, f32)>` ordenado são 24 inline e
//!   uma alocação só.
//! - **SipHash do std**: 2 rounds de ARX por hash. Em 7,67M chaves isso é CPU
//!   real, não só espaço.
//!
//! Resultado medido: **182 bytes por entrada** (1,46x), e o FxHasher também é
//! mais rápido para inferir.

use std::hash::{BuildHasherDefault, Hasher};

/// FxHasher — o mesmo hash que o rustc usa para tabelas de símbolo.
/// Multiplicação e rotação; sem tabela de arredondamento, sem SipHash.
#[derive(Default, Clone)]
pub struct FxHasher {
    hash: u64,
}

impl FxHasher {
    const SEED: u64 = 0x51_7c_c1_b7_27_22_0a_95;
    const ROTATE: u32 = 5;

    #[inline]
    fn add(&mut self, i: u64) {
        self.hash = (self.hash.rotate_left(Self::ROTATE) ^ i).wrapping_mul(Self::SEED);
    }
}

impl Hasher for FxHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        // `as_chunks` seria mais idiomático, mas continua unstable no toolchain.
        #[allow(clippy::chunks_exact_to_as_chunks)]
        for chunk in bytes.chunks_exact(8) {
            self.add(u64::from_le_bytes(chunk.try_into().unwrap()));
        }
        let resto = bytes.len() % 8;
        if resto > 0 {
            let mut buf = [0u8; 8];
            buf[..resto].copy_from_slice(&bytes[bytes.len() - resto..]);
            self.add(u64::from_le_bytes(buf));
        }
    }

    #[inline]
    fn write_u8(&mut self, i: u8) { self.add(i as u64) }
    #[inline]
    fn write_u32(&mut self, i: u32) { self.add(i as u64) }
    #[inline]
    fn write_u64(&mut self, i: u64) { self.add(i) }
    #[inline]
    fn write_usize(&mut self, i: usize) { self.add(i as u64) }

    #[inline]
    fn finish(&self) -> u64 { self.hash }
}

pub type FxBuild = BuildHasherDefault<FxHasher>;
pub type FxHashMap<K, V> = std::collections::HashMap<K, V, FxBuild>;

/// Contexto canônico: `u64` por contexto de até 3 tokens.
///
/// `pack8d` usa 50 bits por token, então ordem 3 são 150 bits e não cabem
/// num `u64` sem colisão. Colidimos de propósito com FxHash: para 8 milhões de
/// chaves em 2^64 o nascimento esperado de colisões é ~1,7e-6 — abaixo da taxa
/// de erro de qualquer disco. A medição em `tests/chave_canonica.rs` verifica
/// o índice do modelo real.
pub type CtxKey = u64;

#[inline]
pub fn chave(ctx: &[u128]) -> CtxKey {
    let mut h = FxHasher::default();
    for &v in ctx {
        h.write_u64(v as u64);
        h.write_u64((v >> 64) as u64);
    }
    h.finish()
}

/// Distribuição de lexemas de um contexto, ordenada por lexema.
///
/// Ordenar permite busca binária em vez de hash. Nos dados do modelo real a
/// mediana é baixa (poucos candidatos por contexto), então a busca binária
/// custa menos que um segundo hash.
#[derive(Debug, Clone, Default)]
pub struct Cands {
    pub items: Vec<(u32, f32)>,
}

impl Cands {
    #[inline]
    pub fn get(&self, lex: u32) -> Option<f32> {
        self.items
            .binary_search_by_key(&lex, |&(l, _)| l)
            .ok()
            .map(|i| self.items[i].1)
    }

    #[inline]
    pub fn get_mut(&mut self, lex: u32) -> Option<&mut f32> {
        let i = self.items.binary_search_by_key(&lex, |&(l, _)| l).ok()?;
        Some(&mut self.items[i].1)
    }

    #[inline]
    pub fn entry_or_default(&mut self) -> &mut Vec<(u32, f32)> {
        &mut self.items
    }

    /// Soma de todos os candidatos.
    #[inline]
    pub fn total(&self) -> f32 {
        self.items.iter().map(|&(_, v)| v).sum()
    }

    #[inline]
    pub fn len(&self) -> usize { self.items.len() }

    #[inline]
    pub fn is_empty(&self) -> bool { self.items.is_empty() }

    /// Insere ou acumula, mantendo a ordem por lexema.
    pub fn add(&mut self, lex: u32, delta: f32) {
        match self.items.binary_search_by_key(&lex, |&(l, _)| l) {
            Ok(i) => self.items[i].1 += delta,
            Err(i) => self.items.insert(i, (lex, delta)),
        }
    }

    /// Une outra distribuição nesta (soma candidato a candidato).
    pub fn merge(&mut self, outra: &Cands, peso: f32) {
        for &(lex, v) in &outra.items {
            self.add(lex, v * peso);
        }
    }
}

impl FromIterator<(u32, f32)> for Cands {
    fn from_iter<I: IntoIterator<Item = (u32, f32)>>(iter: I) -> Self {
        let mut items: Vec<(u32, f32)> = iter.into_iter().collect();
        items.sort_unstable_by_key(|&(l, _)| l);
        Self { items }
    }
}
