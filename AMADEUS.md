# AMADEUS — Arquitetura e Evolução

## O que é

AMADEUS é um **compilador morfológico + modelo tabular de 3 níveis com hipercubo 7D**.
Não é uma rede neural. Não usa retropropagação. Não vê texto — apenas inteiros.

Cada token é um `Token7` com 8 campos inteiros:

| Campo | Tipo | Bits | Função |
|-------|------|------|--------|
| `lex` | u32 | 32 | ID da raiz lexical (compartilhado entre formas flexionadas) |
| `morph` | u16 | 16 | Traços morfológicos: classe(3) + gênero(1) + número(1) + tempo(3) + pessoa(3) |
| `syn_off` | i16 | 16 | Deslocamento negativo para o head na árvore de dependência (0 = raiz) |
| `syn_func` | u8 | 8 | Função sintática: sujeito(0), OD(1), OI(2), ADV(3), root(4), ADN(5), PREP(6), pred(7) |
| `orth` | u8 | 8 | Ortografia (maiúscula/minúscula/pontuação) |
| `punct` | u8 | 8 | Pontuação associada ao token |
| `style` | u16 | 16 | Estilo/afeto (modulado pelo oscilador Collatz) |
| `graph` | u32 | 32 | Embedding semântico via random indexing (Hamming < 16 ≈ similar) |

Um token serializa como **8 valores u32** no plano sequencial (2× u64).

---

## Arquitetura — Visão Geral

```
Texto → Compiler → Vec<Token7> → assign_dependencies() →
  ├─ GRAPH: build_graph() → random indexing u32 por token
  ├─ T3 (morph, syn_func, style) → lex
  ├─ T2 multi-hop (head + avô) → morph_detail, style  
  └─ CUBO 7D (contexto completo) → distribuição sobre próximo lex
       └─ modulação GRAPH: espalha massa entre top-K similares (distância Hamming)

Geração:
  CUBO(histórico 7D + GRAPH) → lex_candidato → T2(head/avô) → morph, style → T3(confirma)
```

### Embedding GRAPH (Random Indexing)

Cada token recebe um `u32` que codifica similaridade semântica via co-ocorrência:

- `build_graph(tokens, window=3)`: para cada token, XOR com rotação à direita dos `graph` vizinhos
- Similaridade = `1 - Hamming(graph_a ^ graph_b) / 32`
- Similaridade > 0.5 → palavras são semanticamente próximas

**Modulação no CUBO**: durante `sample_lex`, o top-20 da distribuição interpolada
tem massa espalhada entre pares com Hamming < 16 (similaridade > 0.5), escalado
por `alpha = 0.15`. Isto permite que sinônimos e palavras relacionadas dividam
probabilidade entre si, mesmo sem co-ocorrência exata no treino.

### Hipercubo 7D (CUBO)

Substitui o T1 (n-grama de POS) por um modelo de **7 dimensões simultâneas**:

```
Cada token do histórico é compactado em u64:
  LEX_HASH(16) | MORPH(16) | SYN_BIN(4) | SYN_FUNC(4) | PUNCT(4) | STYLE(8) | CLAUSE(4)

Tabela: HashMap<Vec<u64>, HashMap<u32, f32>>
  chave = sequência de K tokens compactados
  valor = distribuição de contagens sobre o próximo lex

Interpolação Jelinek-Mercer:
  P(lex | h) = Σ λₙ · Pₙ(lex | h₁..ₙ) + λ_uni · P(lex)
  λₙ = 0.5ⁿ normalizado, λ_uni = 0.10
```

### T2 — Propagação de Traços Morfossintáticos

**Simples**: `(head.morph, head.style, curr_class) → (curr_morph, curr_style)`

**Multi-hop (T2+)**: `(avo.morph, avo.style, curr_class) → (curr_morph, curr_style)`
- Interpola 0.4 avô + 0.6 head quando ambos existem
- Permite concordância através de 2 saltos na árvore (ex: ART → N → ADJ)

### T3 — Seleção Lexical com Cascata

5 níveis de fallback progressivo:
1. `(morph, syn_func, style)` → lex (exato)
2. `(class, syn_func, style)` → lex (ignora detalhe morfológico)
3. `(class, syn_func)` → lex (ignora estilo)
4. `(class, style)` → lex (ignora função sintática)
5. `class` → lex (qualquer palavra da classe)

### Clause Stack

Rastreio de profundidade de subordinação durante treino e geração:
- PREP/ADV subordinadores → incrementa profundidade
- PONT final → decrementa
- Profundidade alimenta a dimensão CLAUSE(4 bits) do CUBO

### Oscilador Collatz

```
n₀ = 27
n+1 = n/2 se n par, 3n+1 se n ímpar
Par (Originalist): temperatura × 0.8, exploração × 0.5
Ímpar (Vanguardist): temperatura × 2.2, exploração × 2.5
```

### Constituição

3 regras imutáveis + 2 emendas mutáveis (votadas pelo Collatz):
1. ART → SUBST (artigo sempre seguido de substantivo)
2. PREP → ¬VERB (preposição nunca antes de verbo)
3. Concordância de gênero ART/ADJ → SUBST

---

## Evolução do Modelo

| Versão | Arquitetura | T1 | T2 | T3 | Dados | tok/s | Tamanho |
|--------|-------------|----|----|----|-------|-------|---------|
| **v0** | byte-level n-gram | — | — | — | ~10KB | — | — |
| **v1** | idgram (id<<8\|pos) | — | — | — | ~10KB | — | — |
| **v2** | T1/POS + T2/128bit + T3/lex (3 tabelas separadas) | 338 | 434 | 105 | ~9K | — | — |
| **v3** | Token7 + T1/T2/T3 reescritos | 212 | 760 | 484 | ~9K | — | — |
| **v4** | +T3 cascata 5 níveis + formato v1 (.grammar7) | 222 | 784 | 486 | ~11K | — | 3.4MB |
| **v5** | +CUBO 7D (substitui T1) + T2 multi-hop + clause stack + formato v2 | **165K** | 805 | 486 | **~73K** | **416** | **6.9MB** |
| **v5.1** | **+GRAPH embedding (random indexing) + modulação semântica no CUBO** | **165K** | 805 | 486 | **~73K** | **416** | **6.9MB** |

### v5.2 — Estado atual

```
CUBO:  174.920 contextos de 7 dimensões
T2:        808 padrões de concordância (head imediato)
T2+:       377 padrões de concordância (multi-hop, avô)
T3:        485 entradas lexicais exatas
T3 (cascata): 1.402 (T3) + 37 (CSS) + 13 (CS) + 19 (CST) + 7 (Class)
GRAPH:   1.682 embeddings u32 (random indexing, window=3)
Compiler: 10.114 entradas no léxico
Lex→Class: ~9.000+ mapeamentos
Underworld: ~1.1M tokens (3.268 arquivos .md em 17 diretórios)
  ├─ 9 domínios semânticos (gerado_semantico/): tecnologia, natureza,
  │  filosofia, agricultura, astronomia, arte, conhecimento, sentimento,
  │  matemática — cada um com vocabulário próprio e padrões sintáticos
  ├─ 10 lotes de geração massiva (gerado_massivo/): templates com
  │  substituição lexical aleatória
  ├─ conhecimento, filosofia, arte, pessoas, natureza, tecnologia:
  │  textos narrativos com [[links]] internos (random walk)
  └─ exercícios de concordância (gerado_semantico/concordancia/)
PCFG: ~9.000 tokens sintéticos por iteração (×10)
Arquivo: ~7MB (grammar) + 66KB (compiler)
Inferência: ~400 tok/s (CPU, single thread)
Clause-stack: geração rastreia profundidade de subordinação
Código: 7.384 linhas Rust (51 arquivos)
  ├─ amadeus_m: 5.180 linhas (28 arquivos)
  ├─ GGUF engine: 1.551 linhas (15 arquivos)
  └─ binários: 653 linhas (6 arquivos)
```

Novo em v5.2:
- **Campos semânticos**: gerador `gerar_campos_semanticos.py` organiza treino em 9 domínios com vocabulário próprio — palavras do mesmo domínio co-ocorrem, ensinando o GRAPH embedding a agrupar conceitos relacionados
- **Underworld expandido**: 75K → 1.1M tokens (~15×), saturando T3 com todas as combinações morfossintáticas possíveis
- **Clause-stack na geração**: `HierarchicalCubo::generate()` rastreia profundidade de subordinação incrementalmente, alimentando a dimensão CLAUSE(4 bits) do hipercubo

---

## Próximos Passos (v5.2)

1. ~~GRAPH Embedding~~ ✅
2. ~~Expansão do léxico~~ ✅
3. ~~Clause-stack na geração~~ ✅
4. ~~GRAPH como dimensão do CUBO~~ ✅
5. ~~Interpolação aprendida~~ ✅
6. ~~Underworld 500K+~~ ✅ — ~1.1M tokens gerados
7. 🔄 **Treino completo + avaliação** — rodar `amadeus_m_train --iterations 10` com o novo underworld
8. **Geração hierárquica com clause-stack** — refinar geração multi-nível usando profundidade real

---

## Comparação com IA Atual

### Versus Transformers (GPT, LLaMA, etc.)

| Característica | AMADEUS v5.1 | Transformer (7B) |
|----------------|------------|-------------------|
| **Aprendizado** | Contagem + interpolação | Retropropagação + gradiente |
| **Parâmetros** | ~600K contagens | 7.000.000.000+ pesos |
| **Dados de treino** | 73K tokens | 2-10T tokens |
| **Representação** | Símbolos inteiros (Token7) | Embeddings contínuos (fp16) |
| **Sintaxe** | Codificada (syn_off + syn_func) | Aprendida estatisticamente |
| **Morfologia** | Raiz + bits de flexão | Tokens independentes |
| **Custo treino** | ~30s CPU | $1M+ GPU |
| **Inferência** | 416 tok/s (1 core CPU) | ~50 tok/s (1 GPU) |
| **Memória** | 7MB | 14GB+ |
| **Generalização** | Estrutural (padrões sintáticos) | Semântica (contexto amplo) |
| **Alucinação** | Baixa (só escolhe palavras do léxico) | Alta (texto fluente mas factualmente errado) |
| **Explicabilidade** | Total (/porque mostra cada passo) | Caixa-preta |
| **Afeto** | Collatz + mood modulam geração | Ausente |
| **Extrapolação** | Fraca (só viu 73K tokens) | Forte (viu bilhões de tokens) |

### Em termos de engenharia

- AMADEUS é **mais rápido**, **menor**, **mais barato** e **completamente interpretável**
- GRAPH adiciona similaridade semântica via random indexing — sem redes neurais
- Transformer é **mais fluente**, **mais flexível** e **generaliza muito melhor**
- AMADEUS capta estrutura linguística **explícita**; Transformer capta **implícita**
- AMADEUS **não escala** (precisão linear com dados); Transformer escala com lei de potência

### Para que AMADEUS serve

- **Análise sintática** de baixo custo (73K tokens → parser funcional)
- **Geração com restrições explícitas** (gênero, número, caso, função sintática)
- **Sistemas embarcados** (roda em 7MB, sem GPU)
- **Aplicações que exigem rastreabilidade** (cada decisão é explicável)
- **Prototipagem rápida** (treino em segundos, iteração imediata)

### Para que NÃO serve

- **Texto livre criativo** — não tem fluência semântica
- **Tradução, sumarização, QA** — não viu dados suficientes
- **Domínios abertos** — léxico fechado de 2.700 palavras
- **Aprendizado de conceitos abstratos** — sem embeddings contínuos

---

## Próximos Passos

1. ~~GRAPH Embedding~~ ✅ — cada lex com `u32` de random indexing; modulação por
   Hamming < 16 no CUBO (top-20, alpha=0.15)
2. ~~Expansão do léxico~~ ✅ — léxico expandido para **10.114 entradas**
3. **Underworld** — 75K → 500K+ tokens para saturar T3
4. **Geração hierárquica por clause-stack** — planejamento de orações antes de tokens
5. **Interpolação aprendida** — λs do CUBO ajustados por EM em dados held-out
6. **GRAPH como dimensão do CUBO** — usar `graph(8 bits)` como 8ª dimensão no
   packing do hipercubo, em vez de apenas modulação pós-hoc
