# AMADEUS Progress Log

## 2026-09-14 — v6.0: SNN Re-architected, Corpus Misto, Phrase Generation

### Resultados
- **SNN re-escrita**: 7 outputs (POS classes) em vez de 50K outputs (lexemas)
- **Acurácia**: Top-1=82.8%, Top-3=86.0% com lr=0.001 (lr=0.01 destruía pesos)
- **Training time**: ~30s para 1.055 exemplos × 10 épocas (vs timeout anterior)
- **STDP**: 10 épocas com Fisher-Yates shuffle, top-1=top-3=86% no pico
- **Corpus misto**: Wikipédia (278 artigos, 359K tokens) + 5 romances Gutenberg (151K tokens) = 510K tokens totais
- **Geração por frazes**: `generate_phrase()` + `generate_text()` — CUBO→classe→T3→GRAPH
- **GRAPH PPMI**: Pointwise Mutual Information weighted random projections substitui random projections puras
- **GRAPH O(n²) → O(n)**: Inverted co-occurrence map, ~2.6ms para 50K tokens
- **Bigramas contextuais**: `context_bigrams`, `context_totals`, `unigrams`, `sample_lex_with_context()`
- **Modulação semântica**: `sample_structured()` — CUBO→classe→GRAPH similar+frequency
- **Memória global**: XOR de embeddings entre frazes
- **Commits**: `c39e2dd` (HMM), `36a87a5` (iterative STDP), `6c07ecb` (GRAPH optimized), `a51900d` (SNN re-arch), `2f5269b` (bigrams), `cbe0b8f` (phrase gen)

### Geração observada

Corpus misto + frazes:
```
a
as palavras que o phytor o pôsto pala
do as que o campo o campo que o campo o corpo do
```

Wikipédia + HMM (wikipedia_article.txt):
```
pronunciada solo gramatura no qual natal descoberta por no ano era ano vinte e dois da
quais formulou se a britânica sistemática biologia taxonomia
segundo em o ano vinte e nove ampla que estabelecia que todo classe dos seis semanas
```

Artigo Didático (artigo_didatico.txt):
```
amplamente proteínas muito metabólicos ditas e que são responsáveis a reações em uma concentrações por
necessárias somente quantidades que possuem transformam outros e outras em outras aceleração produtivas
e que são processos substâncias velocidades de reações em reações das quais e que são diferentes
do catalisam biológico que são
```

### Análise

**O que funciona:**
- HMM: 81% precisão POS (vs ~50% aleatório)
- Estrutura gramatical correta (SUBST→ART, VERB→ADV, etc.)
- Concordância de gênero/número via bias T2/T3
- Phase 1: coocorrência exata ou class-based

**O que não funciona:**
- Phase 2 (similaridade semântica): GRAPH 32-bit muito coarse para similaridade entre palavras
- **Geração estrutural funciona**: artigos antes de substantivos, preposições antes de substantivos, verbos em posições corretas
- **Mas não é coerente**: frequência pura escolhe palavras aleatórias da mesma classe POS

### Próximos passos
1. **Memória global**: XOR de embeddings entre frazes (cross-phrase memory)
2. **Backoff estruturado**: CUBO→classe, T3→classe_similar, GRAPH desempata
3. **Vocabulário reduzido**: top 5K palavras mais frequentes (OOV→UNK)
4. **Avaliação**: BLEU/ROUGE vs corpus de teste

---

## 2026-09-13 — HMM POS Tagger

### Implementação
- **módulo `hmm.rs`**: 505 linhas — Viterbi, treinamento com Laplace smoothing, 200+ seed words
- **Fixes**: balanced priors, expanded verb/adjective seeds, context-counting (ex: "o campo" → class=3)
- **Métricas treino**: ~50% accuracy (balanceada por classes, não por tokens)

---

## 2026-09-12 — GGUF Binary Format v2

### Implementação
- **Estrutura**: 16 tensors, 22 integrity tests pass
- **Dimensions**: GRAPH(4096,32), T3(1055×7) + vocab, T2(805×8), HMM priors(8), W2C embeddings(5120,32)

### Training pipeline (10 phases)
1. Load W2C embeddings from gguf
2. Random projections 32-bit (u32) for each lexeme
3. Train HMM POS tagger
4. HMM guided assignment
5. Compute vocabulary frequencies
6. Build morph tables
7. Build syntax tables
8. Build GRAPH embeddings (random indexing, 32-bit)
9. Build CUBO
10. Train SNN with STDP

### Commits
- `379ee87` — Final GGUF v2 + grammar v3
- `7474234` — SNN integration + OOV fix
- `f985b26` — EmbeddingCompressor + DualLoader + GRAPH interpolation
- `6c1ed98` — Fill Eos/Bos/Unk with first 3 lexemes
- `64d3227` — Fix GGUF serialization, SNN module
- `268f3f1` — SNN module implementation
- `82e9649` — Gramática v3 + Serialização + Atualização
- `2e5379f` — SNN module (snn.rs, lib.rs, TripleGrammar, GGUF writer/reader)
- `d906e30` — Completando treinamento (grammar v3 + weights legacy)

---

## 2026-09-12 — Training Pipeline + Corpus Wikipédia

### Implementação
- **`train_wikipedia.rs`**: Binary → 10 training phases (HMM → CUBO → SNN → STDP → iterative)
- **Wikipedia extraction**: `wikireader` crate, 278 articles, 1.670 KB
- **Narrative corpus**: 5 Gutenberg romances (Iracema, O Guarany, A Pata da Gazela, Quincas Borba, Yayá Garcia) — 1.515 KB
- **Mixed training**: Wikipedia + narrative = 3.185 KB, 510.291 tokens

### Correções
- `syn_off` overflow: `i16::abs()` → `wrapping_abs()` with `i32` cast
- `assign_dependencies()` O(n²) → O(n) per sentence
- SNN: lr=0.01 → lr=0.001 (stable convergence)

### Commits
- `379ee87` — GGUF v2 + grammar v3
- `7474234` — SNN integration + OOV fix
- `f985b26` — EmbeddingCompressor + DualLoader + GRAPH interpolation
- `6c1ed98` — Fill Eos/Bos/Unk with first 3 lexemes
- `64d3227` — Fix GGUF serialization, SNN module
- `268f3f1` — SNN module implementation
- `82e9649` — Gramática v3 + Serialização + Atualização
- `2e5379f` — SNN module (snn.rs, lib.rs, TripleGrammar, GGUF writer/reader)
- `d906e30` — Completando treinamento (grammar v3 + weights legacy)

---

## 2026-09-12 — ATIVADO (TOOLS on)

### Discutido
- Usuário explicou que **SAMBRÓTULAS v3** ≠ LLM — é ortho-intentioned environment (OrthoSIM + Blueprint + JWT + Infraestrutura)
- Crítica: AMADEUS confunde tabela hash com mente humana, mistura HierarchicalCubo com conceitos de consciência
- Decisão: AMADEUS é **engine estatística**, não IA. Clarificado em README.

### Feito
- README SAMBRÓTULAS reescrito com visão clara
- `TOOLS/STATUS.md` atualizado: AMBOS projetos prontos para deploy (S/O não)
- B7宪章 aprovada
- Sistemas restaurados e operacionais
