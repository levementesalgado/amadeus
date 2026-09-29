//! CUBO com spill para disco.
//!
//! A tabela de n-gram-context cresce sem limite durante o treino (7,67 milhões
//! de contextos no corpus da Wikipédia, ~2,9 GB em RAM). Em inferência a tabela
//! é só *lida* — `sample_lex_modulated` faz `order` lookups por token, e uma
//! geração curta toca poucos milhares de entradas.
//!
//! A estratégia é espelhar o external memory training: manter um shard ativo em
//! RAM, descarregar para um arquivo, e trazer de volta sob demanda por mmap. Como
//! inferência nunca reescreve (só `reinforce`, que é treino), a página só é
//! lida uma vez e o kernel pode descartá-la sob pressão de memória.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use memmap2::Mmap;

/// Contexto serializado: seq de u128 + contagem total + candidatos.
type Ctx = Vec<u128>;

/// Formato de um shard:
/// ```text
/// magic   : [u8; 8]   "AMDSHRD1"
/// n_ctx   : u32
/// n_cands : u32         (total de candidatos no shard, para pré-alocar)
/// total_lex: f32
/// por contexto:
///   ctx_len : u32
///   ctx     : [u128; ctx_len]
///   sum     : f32
///   n_cand  : u32
///   (lex    : u32, cnt : f32) * n_cand
/// ```
const MAGIC: &[u8; 8] = b"AMDSHRD1";

/// Estado de um único shard em disco, mapeado em memória.
struct Shard {
    path: PathBuf,
    /// Contexto -> offset do registro dentro do arquivo. Só o índice fica em RAM.
    index: HashMap<Ctx, u64>,
    mmap: Option<Mmap>,
}

impl Shard {
    fn open(path: &Path) -> std::io::Result<Self> {
        let mut data = Vec::new();
        File::open(path)?.read_to_end(&mut data)?;

        if data.len() < 20 || &data[0..8] != MAGIC {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("shard invalido: {}", path.display()),
            ));
        }

        let n_ctx = u32::from_le_bytes(data[8..12].try_into().unwrap()) as usize;
        let n_cand = u32::from_le_bytes(data[12..16].try_into().unwrap()) as usize;
        let _total_lex = f32::from_le_bytes(data[16..20].try_into().unwrap());

        // Índice: lê os headers de cada contexto, sem materializar candidatos.
        let mut index = HashMap::with_capacity(n_ctx);
        let mut off = 20usize;
        for _ in 0..n_ctx {
            if off + 4 > data.len() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "shard truncado no indice",
                ));
            }
            // O offset guardado é o início do registro, ou seja, a posição do
            // próprio ctx_len. Guardamos antes de avançar.
            let record_start = off;
            let ctx_len = u32::from_le_bytes(data[off..off + 4].try_into().unwrap()) as usize;
            off += 4;
            if off + ctx_len * 16 > data.len() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "shard truncado no contexto",
                ));
            }
            // `as_chunks` seria mais idiomático, mas continua unstable no
            // toolchain atual (1.98), então o lint fica desligado aqui.
            #[allow(clippy::chunks_exact_to_as_chunks)]
            let ctx: Ctx = data[off..off + ctx_len * 16]
                .chunks_exact(std::mem::size_of::<u128>())
                .map(|c| u128::from_le_bytes(c.try_into().unwrap()))
                .collect();
            off += ctx_len * 16;
            let _sum = f32::from_le_bytes(data[off..off + 4].try_into().unwrap());
            off += 4;
            let n_c = u32::from_le_bytes(data[off..off + 4].try_into().unwrap()) as usize;
            off += 4;
            off += n_c * 8; // skip candidatos

            index.insert(ctx, record_start as u64);
        }
        debug_assert!(n_cand > 0 || n_ctx == 0);

        Ok(Self { path: path.to_path_buf(), index, mmap: None })
    }

    /// Mapeia o arquivo se ainda não mapeado, e devolve os candidatos de um contexto.
    /// Retorna por valor: os candidatos são materializados num HashMap próprio,
    /// então não há referência válida a se devolver.
    fn lookup(&mut self, ctx: &[u128]) -> Option<(HashMap<u32, f32>, f32)> {
        // Offset guardado aponta para o inicio do registro (ctx_len).
        let &start = self.index.get(ctx)?;

        if self.mmap.is_none() {
            let file = File::open(&self.path).ok()?;
            // SAFETY: o arquivo é imutável apos escrito (shards só são lidos
            // durante inferência; treino sempre grava num shard novo).
            let mmap = unsafe { Mmap::map(&file) }.ok()?;
            self.mmap = Some(mmap);
        }
        let m = self.mmap.as_ref().unwrap();

        let mut off = start as usize;
        let ctx_len = u32::from_le_bytes(m[off..off + 4].try_into().unwrap()) as usize;
        off += 4 + ctx_len * 16;
        let sum = f32::from_le_bytes(m[off..off + 4].try_into().unwrap());
        off += 4;
        let n_c = u32::from_le_bytes(m[off..off + 4].try_into().unwrap()) as usize;
        off += 4;

        // Materializa candidatos num HashMap próprio. Seria ideal evitar, mas a
        // API de `sample_lex_modulated` itera candidatos; o custo é por acesso.
        let mut cands = HashMap::with_capacity(n_c);
        for i in 0..n_c {
            let b = off + i * 8;
            let lex = u32::from_le_bytes(m[b..b + 4].try_into().unwrap());
            let cnt = f32::from_le_bytes(m[b + 4..b + 8].try_into().unwrap());
            cands.insert(lex, cnt);
        }
        Some((cands, sum))
    }
}

/// Escreve um shard com os contextos fornecidos.
fn write_shard(
    path: &Path,
    table: &HashMap<Ctx, HashMap<u32, f32>>,
    totals: &HashMap<Ctx, f32>,
    total_lex: f32,
) -> std::io::Result<()> {
    let file = File::create(path)?;
    let mut w = BufWriter::with_capacity(1 << 20, file);

    w.write_all(MAGIC)?;
    w.write_all(&(table.len() as u32).to_le_bytes())?;
    let n_cand: usize = table.values().map(|c| c.len()).sum();
    w.write_all(&(n_cand as u32).to_le_bytes())?;
    w.write_all(&total_lex.to_le_bytes())?;

    for (ctx, cands) in table {
        w.write_all(&(ctx.len() as u32).to_le_bytes())?;
        for &v in ctx {
            w.write_all(&v.to_le_bytes())?;
        }
        w.write_all(&totals.get(ctx).copied().unwrap_or(0.0).to_le_bytes())?;
        w.write_all(&(cands.len() as u32).to_le_bytes())?;
        for (&lex, &cnt) in cands {
            w.write_all(&lex.to_le_bytes())?;
            w.write_all(&cnt.to_le_bytes())?;
        }
    }
    w.flush()
}

/// Diretório de shards de um CUBO.
pub struct ShardStore {
    dir: PathBuf,
    shards: Vec<Shard>,
}

impl ShardStore {
    /// Abre (ou cria) o diretório de shards.
    pub fn open(dir: impl AsRef<Path>) -> std::io::Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&dir)?;
        Ok(Self { dir, shards: Vec::new() })
    }

    /// Numero de shards carregados.
    pub fn len(&self) -> usize {
        self.shards.len()
    }

    pub fn is_empty(&self) -> bool {
        self.shards.is_empty()
    }

    /// Carrega todos os shards de um diretório, em ordem lexicográfica.
    pub fn open_dir(dir: impl AsRef<Path>) -> std::io::Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().map(|e| e == "cuboshard").unwrap_or(false))
            .collect();
        paths.sort();

        let mut shards = Vec::with_capacity(paths.len());
        for p in paths {
            match Shard::open(&p) {
                Ok(s) => shards.push(s),
                Err(e) => eprintln!("  WARN: shard ignorado {}: {e}", p.display()),
            }
        }
        Ok(Self { dir, shards })
    }

    /// Descarrega um lote de contextos para um shard novo.
    pub fn flush(
        &mut self,
        table: &HashMap<Ctx, HashMap<u32, f32>>,
        totals: &HashMap<Ctx, f32>,
        total_lex: f32,
    ) -> std::io::Result<PathBuf> {
        let path = self.dir.join(format!("{:05}.cuboshard", self.shards.len()));
        write_shard(&path, table, totals, total_lex)?;
        self.shards.push(Shard::open(&path)?);
        Ok(path)
    }

    /// Procura um contexto em todos os shards.
    pub fn get(&mut self, ctx: &[u128]) -> Option<(HashMap<u32, f32>, f32)> {
        for shard in &mut self.shards {
            if let Some(found) = shard.lookup(ctx) {
                return Some(found);
            }
        }
        None
    }

    /// Total de contextos indexados (sem mmap). Só para métricas.
    pub fn total_contexts(&self) -> usize {
        self.shards.iter().map(|s| s.index.len()).sum()
    }

    /// Bytes ocupados pelos arquivos de shard.
    pub fn bytes_on_disk(&self) -> u64 {
        self.shards.iter().filter_map(|s| s.path.metadata().ok()).map(|m| m.len()).sum()
    }
}
