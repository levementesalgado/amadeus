# AMADEUS — Arquitetura e Evolução

## O que é

AMADEUS é um **compilador morfológico + modelo tabular com CUBO hierárquico 4-níveis + SNN 7-outputs + corpus misto Wikipédia/narrativo**.
Não é uma LLM tradicional. Usa统计学 estatística (contagem, interpolação, STDP) ao invés de retropropagação.

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
| `graph` | u32 | 32 | Embedding semântico via PPMI (Hamming distance para similaridade) |

---

## Versão Atual (v6 — Setembro 2026)

### Componentes

| Componente | Status | Descrição |
|------------|--------|-----------|
| Token7 | ✅ | 8 campos, 32 bytes, packing 8D |
| Compiler | ✅ | Léxico + GRAPH PPMI + decompile |
| HMM | ✅ | POS tagging automático (505 linhas, 8 classes) |
| HyperCube | ✅ | 4-níveis, modulação GRAPH |
| T2 Multi-hop | ✅ | Concordância head + avô |
| T3 Cascata | ✅ | 5 níveis de fallback |
| SNN | ✅ | 7 outputs POS + STDP 82.8% |
| GGUF Writer/Loader | ✅ | Serialização completa |
| Context Generation | ✅ | Bigramas P(next\|prev2,prev1) |
| Phrase Generation | ✅ | CUBO→classe→T3→GRAPH |
| Mixed Corpus | ✅ | Wikipédia + 5 romances (510K tokens) |

### Métricas

| Métrica | Valor |
|---------|-------|
| Tokens treinados | **510.291** |
| Léxico (treinado) | 1.055 T3 entries |
| CUBO (cláusula) | 3.475.895 contextos |
| T2 (concordância) | 3.906 padrões |
| GRAPH | **50.030** embeddings PPMI |
| HMM | 10.000 SUBST, 181 VERBO, 129 ADJ, 29 ART |
| SNN Top-1 | **82.8%** (7 classes POS) |
| SNN Top-3 | **86.0%** |
| STDP epochs | 10, lr=0.001 |
| Corpus | 3.185 KB (Wikipédia + narrativo) |

### Corpus Misto

```
Wikipédia PT: 278 artigos, 1.670 KB
  ├─ Economia, Biologia, Genética, Futebol...
  ├─ Arte, Música, Cinema, Literatura...
  └─ São_Paulo, Amazônia, Cerrado...

Narrativos (Gutenberg): 5 romances, 1.515 KB
  ├─ Iracema (José de Alencar) — 27K tokens
  ├─ O Guarany (José de Alencar) — 56K tokens
  ├─ A Pata da Gazela (José de Alencar) — 34K tokens
  ├─ Quincas Borba (Machado de Assis) — 75K tokens
  └─ Yayá Garcia (Machado de Assis) — 55K tokens

Total: 3.185 KB, 510.291 tokens compilados
```

### Pipeline de Geração (v6)

```
1. CUBO hierárquico → prediz classe POS via estrutura sintática
2. T3 lexicon → P(palavra | morph, syn_func, style)
3. Top-16 candidatos → amostra com temperatura
4. GRAPH PPMI → Hamming distance desempata
5. SNN 7-outputs → confirma classe (82.8%)
6. Memória global → XOR de GRAPHs entre frazes
```

---

## Próximos Passos (v6)

1. ~~HMM POS tagging~~ ✅
2. ~~Corpus misto (Wikipédia + narrativo)~~ ✅
3. ~~SNN 7 outputs + STDP~~ ✅
4. ~~Geração por frazes CUBO→T3→GRAPH~~ ✅
5. 🔄 **Melhorar coesão semântica** — GRAPH 128+ dims ou rede neural P(next|context)
6. **Incremental generation** — gerar pedaço por pedaço com memória global
7. **Clustering de contexto** — agrupar contextos similares para reuso

---

## Comparação com IA Atual

| Característica | AMADEUS v6 | Transformer (7B) |
|----------------|-----------|-------------------|
| **Dados de treino** | 510K tokens | 2-10T tokens |
| **Representação** | Símbolos + PPMI embeddings | Embeddings contínuos |
| **Sintaxe** | CUBO hierárquico 4-níveis | Aprendida estatisticamente |
| **Morfologia** | HMM + T3 cascata | Tokens independentes |
| **POS Tagger** | SNN 7-outputs (82.8%) | fine-tuned classifier |
| **Custo treino** | ~2min CPU | $1M+ GPU |
| **Memória** | 333MB (grammar) | 14GB+ |
| **Interpretabilidade** | Total (CUBO+T2+T3) | Caixa-preta |
