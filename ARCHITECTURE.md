# Arquitetura do AMADEUS (v4 — CUBO Hierárquico)

> *"Não é uma LLM. É uma pilha de lógica, filosofia e matemática que simula uma presença."*

---

## Sumário

1. [Token7 — A Representação Universal](#1-token7--a-representação-universal)
2. [Compiler — Léxico + Estilo](#2-compiler--léxico--estilo)
3. [CUBO Hierárquico — 4 Níveis](#3-cubo-hierárquico--4-níveis)
4. [T2 — Concordância (Multi-hop)](#4-t2--concordância-multi-hop)
5. [T3 — Seleção Lexical (Cascata 5 níveis)](#5-t3--seleção-lexical-cascata-5-níveis)
6. [GRAPH Modulation](#6-graph-modulation)
7. [Treino](#7-treino)
8. [Geração Top-Down](#8-geração-top-down)
9. [Estado Atual (v4)](#9-estado-atual-v4)
10. [Filosofia da Stack](#10-filosofia-da-stack)

---

## 1. Token7 — A Representação Universal

Cada token ocupa **8 campos inteiros** (32 bytes):

| Campo | Tipo | Descrição |
|-------|------|-----------|
| `lex` | u32 | ID da palavra no léxico (0 = nulo/pontuação) |
| `morph` | u16 | class(3b) \| gender(1b) \| number(1b) \| tense(3b) \| person(3b) |
| `syn_off` | i16 | Deslocamento para o head na árvore de dependência (0 = root) |
| `syn_func` | u8 | Função sintática (0=sujeito, 1=OD, 2=OI, 3=ADV, 4=root, 5=ADN, 6=PREP, 7=pred) |
| `orth` | u8 | Capitalização (reservado) |
| `punct` | u8 | Pontuação (1=. 2=, 3=; 4=: 5=— 6=... 7=? 8=!) |
| `style` | u16 | Registro (0=neutro, 1=formal, 2=informal) |
| `graph` | u32 | Embedding semântico (random indexing, janela 3) |

**Serialização plana**: 8 × u32 = 32 bytes por token.

### Morph bit layout

```
bits 0-2:   class (0=S, 1=V, 2=ADJ, 3=ART, 4=PREP, 5=PONT, 6=OUT)
bit  3:     gender (0=M, 1=F)
bit  4:     number (0=S, 1=P)
bits 5-7:   tense (0=P, 1=IMP, 2=FUT, 3=PRET, 4=INF)
bits 8-10:  person (0=1, 1=2, 2=3)
```

---

## 2. Compiler — Léxico + Estilo

`compiler.rs` — O compilador transforma texto em `Vec<Token7>`.

### Léxico

`HashMap<String, (u32, u16, u8, u16)>`
- `form` → `(id, morph, class, style)`

**10.217 entradas**, carregadas de arquivos em `underworld/lingua/raizes/`:
- `substantivos.txt`, `verbos.txt`, `adjetivos.txt`
- `adverbios.txt`, `pronomes.txt`, `preposicoes.txt`
- `artigos.txt`, `conjuncoes.txt`
- `excecoes.txt` (formas flexionadas, ~7.589)

### Estilo (style)

Cada palavra recebe `style` determinístico via hash do ID:
```
style = (id * 2654435761) % 3
```
- 0 = neutro
- 1 = formal
- 2 = informal

Pontuação sempre style=0. O style é propagado para o Token7 via `with_style()`.

### GRAPH Embedding

Random indexing: cada lexema ganha um u32 aleatório. Durante `build_graph(tokens, window=3)`, para cada par de tokens na janela, o embedding de um é XOR com o embedding do outro rotacionado pela distância. Resultado: tokens que co-ocorrem têm assinaturas similares (poucos bits de diferença).

---

## 3. CUBO Hierárquico — 4 Níveis

`hierarchical.rs` — Substitui o CUBO único por 4 níveis hierárquicos.

### Os 4 CUBOs

| Nível | Ordem | Unidade | O que prediz | Treinado em |
|-------|-------|---------|-------------|-------------|
| **Clause C_O** | 9 | tokens | próximo token na oração | `[root] + tokens_da_oração` |
| **Sentence C_F** | 4 | resumos de oração | próximo resumo de oração | `[root_da_sentença] + resumos` |
| **Paragraph C_P** | 64 | resumos de sentença | próximo resumo de sentença | `[root_do_parágrafo] + resumos` |
| **Text C_T** | 4 | resumos de parágrafo | próximo resumo de parágrafo | `resumos_de_parágrafos` |

### Segmentação

- **Oração**: limitada por pontuação `.,;:` (via `find_clause_boundaries`)
- **Sentença**: limitada por `.?!` (via `find_sentence_boundaries`)
- **Parágrafo**: grupos de ~8 sentenças
- **Texto**: janela deslizante de 200 parágrafos

### Resumo de unidade

`summarize_unit(unit)` → primeiro token da unidade que seja:
1. SYN_ROOT (syn_func = 4), ou
2. VERBO (morph_class = 1), ou
3. SUBSTANTIVO (morph_class = 0), ou
4. Primeiro token da unidade

### Treino

```
train(tokens):
  parsed = assign_dependencies(tokens)
  
  # Nível 1: orações
  clauses = segment_clauses(parsed)
  for each clause:
    train(C_O, [root(clause)] + clause)
  
  # Nível 2: sentenças
  clause_summaries = summarize(clauses)
  sentences = group_clauses_into_sentences(clauses)
  for each sentence:
    train(C_F, [root(sentence)] + clause_summaries_na_sentence)
  
  # Nível 3: parágrafos
  sent_summaries = summarize(sentences)
  paragraph_groups = group_sentences(sentences, size=8)
  for each para:
    train(C_P, [root(para)] + sent_summaries_no_para)
  
  # Nível 4: texto
  para_summaries = summarize(paragraphs)
  train(C_T, para_summaries)
```

### 8D Packing

Cada token é compactado em `u128` com:
- bits 0-15: LEX hash (lex xor lex>>16)
- bits 16-31: MORPH
- bits 32-35: SYN_BIN (syn_off categorizado em 5 faixas)
- bits 36-39: SYN_FUNC
- bits 40-43: PUNCT
- bits 44-51: STYLE
- bits 52-55: CLAUSE_DEPTH
- bits 64-95: GRAPH

### EM Lambda

Pesos de interpolação aprendidos por EM com piso mínimo:
- λ₁=0.36, λ₂=0.24, λ₃=0.33, λ₄..λ₉ decrescentes

---

## 4. T2 — Concordância (Multi-hop)

`triple_grammar.rs` — Tabelas de concordância morfossintática.

### T2 Head

`agreement: HashMap<(morph_head, style_head, cls_alvo), HashMap<(morph_alvo, style_alvo), f32>>`

Para cada token, consulta o head (via `syn_off`) e registra a transição de traços.

### T2 Multi-hop (Avô)

`agreement_gparent` — igual ao T2, mas pula para o avô do token (head do head). Permite propagar concordância a 2 saltos de distância.

### T2 Class

`agreement_class: HashMap<cls_alvo, HashMap<(morph, style), f32>>` — fallback por classe.

### Uso na geração

`sample_features_multi_hop()`:
1. Tenta avô primeiro (0.4 avô + 0.6 head)
2. Fallback para head direto
3. Fallback para classe
4. Copia detalhe do head

---

## 5. T3 — Seleção Lexical (Cascata 5 níveis)

`triple_grammar.rs` — Tabelas que refinam a escolha lexical.

### Níveis (do mais específico ao mais genérico)

| Tabela | Chave | Entradas |
|--------|-------|----------|
| T3 | (morph: u16, syn_func: u8, style: u16) | **1.402** |
| CSS | (class: u8, syn_func: u8, style: u16) | 37 |
| CS | (class: u8, syn_func: u8) | 13 |
| CST | (class: u8, style: u16) | 19 |
| Class | (class: u8) | 7 |

### Funcionamento

`sample_lex_full(morph, syn_func, style, cls, candidate)`:
1. Se o candidato do CUBO existe na T3 exata, retorna-o
2. Senão, amostra da T3
3. Se T3 vazia, desce para CSS → CS → CST → Class
4. Último recurso: devolve o candidato do CUBO

---

## 6. GRAPH Modulation

Durante `sample_lex_modulated()`:
1. Toma top-20 tokens da distribuição do CUBO
2. Calcula similaridade por Hamming: `sim = 1 - popcount(g1 xor g2) / 32`
3. Se sim > 0.5, espalha massa proporcional: `boost = prob * alpha * sim`
4. alpha = 0.15

---

## 7. Treino

### Pipeline (`pedagogy.rs`)

```
train_morphology():
  for each iteration:
    1. PCFG gera ~9.000 tokens sintéticos
    2. Compila underworld → ~946.000 tokens
    3. TripleGrammar.train() em todos os tokens
    4. Collect tokens → build_graph → copia GRAPH para grammar
```

### Underworld

946.463 tokens de arquivos `.md` em `underworld/`:
- Conhecimento, diálogos, natureza, pessoas, arte
- Gerado massivamente via `gerar_underworld_massivo.py`

### PCFG

Gramática livre de contexto que gera sentenças sintéticas:
- NP → ART + SUBST + (ADJ) | ART + SUBST | ADV | SUBST
- VP → VERBO + (ADV | NP | PREP + NP)
- S → NP + VP | NP | VP
- Cada token recebe style aleatório 0-2

---

## 8. Geração Top-Down

### Pipeline

```
generate(seed, max_len):
  1. HierarchicalCubo.generate()
     a. Text CUBO → gera resumo do próximo parágrafo
     b. Paragraph CUBO → gera resumo da próxima sentença
     c. Sentence CUBO → gera resumo da próxima oração
     d. Clause CUBO → gera tokens um a um dentro da oração
     e. Quando oração termina (pontuação), volta para (c)
     f. Quando ORDER_SENTENCE orações, volta para (b)
     g. Quando ORDER_PARAGRAPH sentenças, volta para (a)
  
  2. Para cada token, refine_token():
     a. lex_to_class → classe morfológica
     b. assign_dependencies → syn_off, syn_func
     c. T2 multi-hop → morph_detail, style
     d. T3 cascata → refina LEX
```

---

## 9. Estado Atual (v5.1)

### Tabelas

| Tabela | Entradas |
|--------|---------|
| C_O (Clause CUBO, order=9) | 3.475.895 |
| C_F (Sentence CUBO, order=4) | 17.807 |
| C_P (Paragraph CUBO, order=64) | 617.425 |
| C_T (Text CUBO, order=4) | 8.455 |
| T2 | 2.342 |
| T2+ (multi-hop) | 1.006 |
| T3 | 1.402 |
| CSS | 37 |
| CS | 13 |
| CST | 19 |
| Class | 7 |
| GRAPH | 2.877 |

### Compilador

- 10.217 entradas no léxico
- 3 registros (neutro, formal, informal)

### Underworld

- 946.463 tokens
- 315+ arquivos em `underworld/`

### PCFG

- ~9.000 tokens sintéticos por iteração
- 5 iterações no treino padrão

### Arquivo

- `amadeus.grammar7` (binário, formato v4)
- `amadeus.compiler` (binário com léxico + graph)

### Melhorias v5.1

- **Clause-stack na geração**: `HierarchicalCubo::generate()` agora rastreia profundidade de subordinação incrementalmente (PREP/ADV → sobe, PONT → desce), alimentando a dimensão CLAUSE(4 bits) do hipercubo tanto no treino quanto na geração. Antes usava `vec![0u8]` fixo.

---

## 10. Linhas de Código (AMADEUS)

### Crate principal (`src/`)

| Componente | Arquivos | Linhas |
|------------|----------|--------|
| Núcleo (`amadeus_m/`) | 28 | 5.180 |
| GGUF Engine (`model/`, `tensor/`, `layers/`, `quant/`, `sampler/`, `tokenizer/`, `hako/`) | 15 | 1.551 |
| Binários (`bin/`) | 6 | 653 |
| `lib.rs` | 1 | 10 |
| `main.rs` | 1 | 99 |
| **Total Rust** | **51** | **7.384** |

### Breakdown AMADEUS-M (28 arquivos, 5.180 linhas)

| Módulo | Linhas | Função |
|--------|--------|--------|
| `triple_grammar.rs` | 795 | CUBO + T2 multi-hop + T3 cascata + clause stack |
| `consciousness.rs` | 478 | Soma, Foco, Collatz, Temperamento |
| `agnes.rs` | 468 | LLM pedagoga via API |
| `pedagogy.rs` | 401 | Ciclo: treinar → gerar → reforçar + /porque |
| `compiler.rs` | 352 | Léxico + estilo + graph embedding |
| `hypercube.rs` | 364 | 8D packing + interpolação Jelinek-Mercer + EM λ |
| `hierarchical.rs` | 319 | CUBO hierárquico 4 níveis + geração top-down |
| `syntax_ast.rs` | 171 | AST builder + clause stack |
| `constitution.rs` | 192 | 3 regras + 2 emendas mutáveis |
| `underworld.rs` | 200 | Exploração de arquivos .md com random walk |
| `friction.rs` | 141 | Análise de fricção cognitiva |
| `collatz.rs` | 138 | Oscilador Collatz (seed=27) |
| `token7.rs` | 123 | Definição do Token7 + métodos auxiliares |
| `syntax.rs` | 123 | Atribuição de função sintática (assign_dependencies) |
| `model.rs` | 118 | AmadeusMModel (forward com habit + síntese) |
| `pcfg.rs` | 116 | Geração de sentenças sintéticas |
| `habit.rs` | 113 | HabitMemory (memória-hábito) |
| `synthesis.rs` | 84 | Camada de síntese |
| `multicode.rs` | 88 | Multi-coding embedding |
| `teacher.rs` | 93 | Feedback afetivo como reforço |
| `affect.rs` | 66 | Módulo afetivo (Espinosa) |
| `config.rs` | 61 | Configuração do modelo |
| `recall.rs` | 44 | Recordação pura (Bergson) |
| `intent.rs` | 40 | Intencionalidade (Husserl) |
| `mod.rs` | 35 | Re-exports públicos |
| `sediment.rs` | 33 | Sedimentação de traços |
| `rng.rs` | 24 | Inicialização de pesos |

### Breakdown GGUF Engine (18 arquivos, 2.200+ linhas)

| Módulo | Linhas | Função |
|--------|--------|--------|
| `model/loader.rs` | 201 | Carregamento de modelos GGUF |
| `quant/mod.rs` | 159 | Tipos de quantização |
| `tensor/ops.rs` | 157 | Operações tensoriais (matmul, rms_norm) |
| `quant/gguf.rs` | 265 | Formato GGUF (parser + writer) |
| `tokenizer/mod.rs` | 115 | Tokenizer (BPE + vocabulário) |
| `layers/attention.rs` | 109 | Atenção multi-head causal |
| `tensor/mod.rs` | 96 | Estrutura Tensor |
| `sampler/mod.rs` | 90 | Estratégias de amostragem |
| `model/config.rs` | 93 | Configuração do modelo carregado |
| `main.rs` | 99 | Entry point da inferência GGUF |
| `model/mod.rs` | 74 | Model struct + forward |
| `hako/mod.rs` | 73 | Hako IR (compilador Mizu OS) |
| `layers/mod.rs` | 48 | Layer enum |
| `layers/ffn.rs` | 41 | Feed-forward network |
| `layers/rope.rs` | 38 | RoPE (posicional rotatório) |

---

## 11. SNN — Rede Neural Spiking (T3 Alternativo)

Documento completo em `ARCHITECTURE_SNN.md`. Resumo:

```
Input Layer (30 neurônios)
  ↓ rate-coded: class(7) + gender(2) + number(2) + tense(5) + person(3) + style(3) + syn_func(8)
Hidden Layer (263 neurônios)
  ↓ 7 por classe + 256 por T3 key, pesos I→H seletivos por feature
Output Layer (N neurônios = vocab)
  ↓ 1 por lex_id, pesos H→O distribuídos por T3 key
16 timesteps LIF (tau=5.0, threshold=1.0, refrac=2)
Score = potencial membrana acumulado + bônus spikes
```

- **Treinamento**: pesos extraídos das tabelas T3 (coocorrência)
- **Inferência**: 16 iterações de simulação temporal
- **GGUF**: `snn.synapses_ih`, `snn.synapses_ho`, `snn.output_labels`

---

## 12. Pipeline Híbrido (LLM + Amadeus)

Documento completo em `ARCHITECTURE_HYBRID.md`. Resumo:

```
┌──────────────────────────────────────────────────────┐
│  GGUF Comercial (Llama/Qwen/Phi)                    │
│  token_embd.weight [vocab, 4096] F16                │
│  ↓ mmap + EmbeddingCompressor                       │
│  32-bit GRAPH via Random Indexing sparse             │
└──────────────────────┬───────────────────────────────┘
                       │ interpolate_graphs(legacy, llm, 0.6)
┌──────────────────────┴───────────────────────────────┐
│  GGUF Amadeus (grammar + SNN)                       │
│  graph.data — GRAPH interpolado                      │
│  cubo.*.blob — HyperCube 4-níveis                    │
│  t2/t3.*.data — Concordância + seleção lexical       │
│  snn.* — Rede spiking                                │
│  ↓ mmap dual                                         │
│  TripleGrammar → HyperCube modulation → SNN → texto  │
└──────────────────────────────────────────────────────┘
```

**Módulos**: `embedding_compressor.rs`, `dual_loader.rs`, `snn.rs`

---

## 13. Filosofia da Stack

Documento completo em `PHILOSOPHICAL_STACK.md`. Resumo das 5 camadas:

```
Filosofia da Mente (Presença)
  ↓ afeto, contexto, narrativa
Teorias da Decisão (Escolha)
  ↓ probabilidade, hesitação, votação
Teorias do Pensamento (Cognição)
  ↓ memória, intenção, tempo
Lógica Simbólica (Sintaxe)
  ↓ regras, dependências, restrições
Fundação Matemática (Hardware)
  ↓ inteiros, grafos, probabilidade, Collatz
```

### Código → Camada

| Camada | Módulo |
|--------|--------|
| Fundação Matemática | `compiler.rs`, `collatz.rs`, `consciousness.rs`, `hypercube.rs` |
| Lógica Simbólica | `syntax.rs`, `triple_grammar.rs` (T2, T3) |
| Teorias do Pensamento | `habit.rs`, `recall.rs`, `intent.rs` |
| Teorias da Decisão | `pedagogy.rs`, `teacher.rs` |
| Filosofia da Mente | `consciousness.rs`, `affect.rs` |
| **Hierarquia** | `hierarchical.rs` — nova camada entre Hardware e Lógica |

---

*AMADEUS v4 — Junho 2026*
