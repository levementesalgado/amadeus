# AMADEUS Hybrid — Pipeline LLM + Amadeus

> Mapear pesos de modelos comerciais (Llama, Qwen, Phi) para a lógica do Hipercubo 7D/8D e da SNN.

---

## Sumário

1. [Visão Geral](#1-visão-geral)
2. [Compressor de Embeddings (LLM → GRAPH)](#2-compressor-de-embeddings-llm--graph)
3. [Interpolação de GRAPHs](#3-interpolação-de-graphs)
4. [Dual-GGUF Loader (mmap)](#4-dual-gguf-loader-mmap)
5. [Pipeline de Inferência](#5-pipeline-de-inferência)
6. [Arquivos](#6-arquivos)
7. [GGUF Híbrido](#7-gguf-híbrido)
8. [Benchmark](#8-benchmark)

---

## 1. Visão Geral

```
┌──────────────────────────────────────────────────────────┐
│  GGUF Comercial (Llama 3 / Qwen 2 / Phi-3)             │
│  ┌────────────────────────────────────────────┐          │
│  │ token_embd.weight [128256, 4096] F16      │          │
│  │ (apenas mmap, sem carregar tudo)           │          │
│  └────────────────────┬───────────────────────┘          │
│                       │                                  │
│              EmbeddingCompressor                         │
│              (4096D F16 → 32-bit GRAPH)                  │
│              Random Indexing sparse projections           │
└───────────────────────┬──────────────────────────────────┘
                        │
                        ▼
         interpolate_graphs(legacy, llm, weight=0.6)
                        │
                        ▼
┌───────────────────────┴──────────────────────────────────┐
│  GGUF Amadeus (grammar + SNN)                           │
│  ┌────────────────────────────────────────────┐          │
│  │ graph.data       — GRAPH interpolado       │          │
│  │ cubo.*.blob      — HyperCube 4-níveis      │          │
│  │ t2/t3.*.data     — Concordância + T3       │          │
│  │ snn.*            — Rede spiking             │          │
│  └────────────────────┬───────────────────────┘          │
│                       │                                  │
│              TripleGrammar                              │
│              sample_token()                              │
│                       │                                  │
│              HyperCube.sample_lex_modulated()            │
│              (GRAPH → Hamming spread → probs)            │
│                       │                                  │
│              SNN.infer_snn()                             │
│              (30→263→N, 16 timesteps)                    │
│                       │                                  │
│                    Token7 → texto                        │
└──────────────────────────────────────────────────────────┘
```

### Por que funciona

| Aspecto | LLM Comercial | Amadeus Híbrido |
|---------|---------------|-----------------|
| Sintaxe | Bilhões de parâmetros, attention | T2/T3 tabelas, determinístico |
| Semântica | Embeddings densos (4096D) | GRAPH 32-bit (Hamming) |
| Concordância | KV cache + attention | T2 multi-hop, O(1) |
| Seleção lexical | Softmax sobre vocab inteiro | T3 cascata + SNN |
| Energia | GPU, watts | CPU, mmap |
| Contexto | janela limitada | CUBO 4-níveis, hierárquico |

---

## 2. Compressor de Embeddings (LLM → GRAPH)

### Algoritmo

```
Input: embedding F32[4096] de um token
Output: GRAPH u32 (32 bits)

Para cada bit b em 0..32:
  1. Seed = token_id × 2654435761 + b × 0x9E3779B9
  2. Subamostragem esparsa: ~sqrt(4096) ≈ 64 componentes
  3. Para cada componente i (step = dim/64):
     - sinal = pseudo_aleatório(i, seed) → +1 ou -1
     - sum += embedding[i] × sinal
  4. bit_b = (sum > 0) ? 1 : 0

Return: bits 31..0
```

### Propriedades

- **Determinístico**: mesmo embedding → mesmo GRAPH (seed fixa)
- **Locality-sensitive hashing**: embeddings similares → GRAPH com poucos bits diferentes
- **Sparse**: apenas ~64 de 4096 componentes são usados
- **Rápido**: O(dim × 32 / 64) = O(dim/2) operações

### Exemplo de compressão

```
Llama 3 embedding (4096D F16):
  [0.12, -0.34, 0.56, ..., -0.78]  (4096 valores)

Compressor (seed=token_id):
  bit 0: Σ(embedding[i] × ±1, i=0,64,128,...) = +2.3 → bit=1
  bit 1: Σ(embedding[i] × ±1, i=1,65,129,...) = -0.8 → bit=0
  ...
  bit 31: ... → bit=1

Resultado: 0b10110100...0110 (32 bits)
```

---

## 3. Interpolação de GRAPHs

Quando temos GRAPH legado (treinado no corpus) e GRAPH LLM (compressão semântica):

```rust
pub fn interpolate_graphs(
    legacy: &HashMap<u32, u32>,   // GRAPH do Amadeus
    llm: &HashMap<u32, u32>,      // GRAPH comprimido do LLM
    weight: f32,                   // 0.0 = só legado, 1.0 = só LLM
) -> HashMap<u32, u32>
```

### Lógica

```
Para cada lex_id:
  Se existe em ambos:
    Para cada bit b em 0..31:
      merged_bit = (weight > 0.5) ? llm_bit : legacy_bit
    merged = bits_merged
  Se só existe em legado:
    merged = legacy
  Se só existe no LLM:
    merged = llm
```

### Peso recomendado

- `weight = 0.6`: favorece semântica do LLM (mais generalização)
- `weight = 0.4`: favorece corpus treinado (mais específico)
- `weight = 0.5`: equilíbrio

---

## 4. Dual-GGUF Loader (mmap)

### Uso

```rust
let loader = DualGgufLoader::new(
    "grammar.grammar.gguf",         // GGUF do Amadeus
    Some("llama-3-8b.gguf"),         // GGUF comercial (opcional)
);

let result = loader.load()?;

// result.merged_graph → GRAPH interpolado
// result.lexicon → léxico do compiler
// result.lex_to_class → mapeamento lex→classe
```

### mmap

Ambos os arquivos são abertos via `memmap2::Mmap`:
- **Zero cópia**: dados ficam na memória virtual do OS
- **Lazy loading**: apenas páginas acessadas são carregadas do disco
- **Dual access**: grammar e modelo comercial simultâneos
- **CPU-only**: sem GPU, sem cópias de buffer

### GgufMmapReader

Reader genérico para acessar tensores específicos:

```rust
let reader = GgufMmapReader::open("model.gguf")?;
let embd: Vec<f32> = reader.read_tensor_f32("token_embd.weight")?;
let raw: &[u8] = reader.read_tensor_raw("token_embd.weight")?;
let tensors = reader.list_tensors();
```

---

## 5. Pipeline de Inferência

```
1. Tokenizar texto de entrada
2. Compiler.compile(texto) → Vec<Token7>
   - Cada Token7 tem lex_id, morph, style, graph
   - graph vem do GRAPH interpolado (LLM + legado)

3. Para cada posição i:
   a. HyperCube.sample_lex_modulated(history, clause_depths, graph)
      - Busca n-gramas no CUBO (até ordem 9)
      - Calcula distribuição de candidatos
      - GRAPH modulation: espalha probabilidade entre palavras
        semanticamente similares (Hamming distance < 0.5)
      - Retorna distribuição ponderada

   b. T2.sample_features_multi_hop(histórico, dependências)
      - Propaga concordância via multi-hop
      - Interpola grandparent (40%) + head (60%)

   c. SNN.infer_snn(morph, syn_func, style)
      - Encoding: morph → 30 spike trains
      - 16 timesteps LIF simulation
      - Score: potencial acumulado + spikes
      - Retorna ranking de lex_ids

   d. Selecionar Token7 com maior score
   e. Adicionar ao histórico

4. Gerar texto a partir dos tokens selecionados
```

---

## 6. Arquivos

| Arquivo | Linhas | Descrição |
|---------|--------|-----------|
| `amadeus_m/embedding_compressor.rs` | 180 | Compressor LLM → GRAPH 32-bit |
| `amadeus_m/dual_loader.rs` | 210 | Dual-GGUF loader com mmap |
| `amadeus_m/snn.rs` | 478 | Rede neural spiking |
| `amadeus_m/triple_grammar.rs` | 1471 | TripleGrammar (CUBO+T2+T3+SNN+GRAPH) |
| `amadeus_m/hypercube.rs` | 364 | HyperCube com modulação GRAPH |
| `amadeus_m/compiler.rs` | 352 | Compilador morfológico |
| `bin/test_hybrid.rs` | - | Teste do pipeline híbrido |
| `bin/test_snn.rs` | - | Teste SNN vs cascade |
| `bin/test_gguf.rs` | - | Teste GGUF roundtrip |

---

## 7. GGUF Híbrido

O GGUF final do Amadeus contém tudo:

### Tensors (19+)

| Tensor | Shape | Origem |
|--------|-------|--------|
| `lexicon.forms` | [N, 8] | Compiler |
| `lexicon.roots` | [N] | Compiler |
| `graph.data` | [N, 2] | **Interpolado (LLM + legado)** |
| `lex_to_class.data` | [N, 2] | Compiler |
| `t2.head.data` | [N, 6] | T2 concordância |
| `t2.gparent.data` | [N, 6] | T2 multi-hop |
| `t2.class.data` | [N, 5] | T2 fallback |
| `t3.exact.data` | [N, 5] | T3 exato |
| `t3.css.data` | [N, 5] | T3 class+syn+style |
| `t3.cs.data` | [N, 4] | T3 class+syn |
| `t3.cst.data` | [N, 4] | T3 class+style |
| `t3.class.data` | [N, 3] | T3 classe |
| `cubo.clause.blob` | [N] | HyperCube clause |
| `cubo.sentence.blob` | [N] | HyperCube sentence |
| `cubo.paragraph.blob` | [N] | HyperCube paragraph |
| `cubo.text.blob` | [N] | HyperCube text |
| `snn.synapses_ih` | [30, 263] | SNN pesos I→H |
| `snn.synapses_ho` | [263, N] | SNN pesos H→O |
| `snn.output_labels` | [N] | SNN lex_ids |

### KV Metadata

| Chave | Tipo | Descrição |
|-------|------|-----------|
| `general.architecture` | String | "amadeus" |
| `amadeus.format_version` | U32 | 4 |
| `amadeus.syntax_order` | U32 | 3 |
| `amadeus.temperature` | F32 | 1.0 |
| `amadeus.exploration_rate` | F32 | 0.08 |
| `amadeus.graph_alpha` | F32 | 0.25 |

---

## 8. Benchmark

### Resultados (dados sintéticos, 1500 lex_ids)

| Métrica | Só legado | Híbrido (LLM+legado) |
|---------|-----------|----------------------|
| SNN top-1 | 23.3% | **30.0%** |
| SNN top-3 | 40.0% | **50.0%** |
| Cascade top-1 | 100% | 100% |
| GGUF size | ~390 KB | ~6.2 MB |
| GGUF roundtrip | ✅ | ✅ |

### Latência estimada (T410, mmap)

| Componente | Latência |
|------------|----------|
| mmap grammar | ~0 (lazy) |
| mmap LLM embeddings | ~0 (lazy) |
| GRAPH interpolation | <1 ms |
| HyperCube lookup | <1 ms |
| SNN inference (16 timesteps) | ~10 ms |
| **Total por token** | **~12 ms** |

---

*AMADEUS Hybrid — Setembro 2026*
