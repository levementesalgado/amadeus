# AMADEUS v6.1 — Coesão Semântica + Vocabulário Morfológico

> Sistema de geração de texto com coesão semântica via topic tracking e vocabulário morfológico de afixos.

---

## Visão Geral

```
Texto → Compiler → Vec<Token7> → HMM POS Tagger →
  ├─ GRAPH: morfologia (XOR de afixos) + PPMI (co-ocorrência)
  ├─ T3: seleção lexical (morph, syn_func, style)
  ├─ T2: concordância (head + avô)
  ├─ CUBO 7D: estrutura sintática
  ├─ SNN: classificação POS (7 outputs, 82% acurácia)
  └─ Topic Tracking: memória semântica entre frazes

Geração:
  CUBO → classe → T3 → score(α*coesão + (1-α)*frequência) → GRAPH desempata
```

---

## 1. Vocabulário Morfológico

### Componentes (613 total)

| Tipo | Quantidade | Exemplos |
|------|------------|----------|
| Prefixos | 82 | anti, auto, bi, co, com, con, contra, de, des, dis, en, ex, extra, hiper, im, in, inter, macro, micro, mono, multi, neo, pós, pré, pro, re, retro, semi, sub, super, trans, tri, uni, vice, aero, astro, bio, geo, foto, video, eletro, mega, giga, tera |
| Sufixos | 128 | ção, dade, ismo, mente, oso, ível, ar, er, ir, ando, endo, indo, ado, ido, inho, ito, éta, ote,ão, ona, arra, able, ible, osa, oso, ento, mento, nal, ico, vel, íssimo |
| Raízes | 403 | ser, estar, ter, fazer, dizer, ir, vir, dar, ver, saber, poder, querer, ficar, passar, achar, trazer, casa, vida, mundo, terra, homem, mulher, bom, mau, grande, pequeno, sempre, nunca |

### Decomposição

```
invisível  = in + - + ível      (prefixo + raiz + sufixo)
desconhecido = des + conhec + ido (prefixo + raiz + sufixo)
correndo   = co + - + endo       (prefixo + raiz + sufixo)
brasileiro = - + - + eiro        (raiz + sufixo)
bonitas    = - + - + itas        (raiz + sufixo)
```

### Embeddings

Cada afixo → 32-bit hash determinístico:
- Bits 0-5: comprimento (0-63)
- Bits 6-13: primeira letra (a-z)
- Bits 14-21: última letra
- Bits 22-25: número de vogais
- Bits 26-29: número de consoantes
- XOR com FNV hash

**Palavra = XOR(bits_prefixo, bits_raiz, bits_sufixo)**

Exemplo: "invisível" = `hash(in) ⊕ hash(ível)`

### Similaridade Morfológica

```rust
similarity = (prefixo_match × 0.25 + raiz_match × 0.50 + sufixo_match × 0.25)
```

Palavras com mesma raiz: similarity ≥ 0.50
Palavras com mesmo sufixo: similarity ≥ 0.25

---

## 2. Topic Tracking

### Campos

```rust
pub topic_graph: u32,        // vetor tópico atual
pub topic_momentum: f32,     // inércia (0.8 = muda devagar)
pub recent_graphs: Vec<u32>, // janela de 32 tokens
pub cohesion_alpha: f32,     // peso coesão vs frequência (0.4)
```

### Algoritmo

```
1. Para cada token gerado:
   - Adicionar GRAPH à janela recente
   - Calcular média ponderada exponencial
   - Misturar com tópico anterior (momentum)

2. Momentum: topic_graph = 80% anterior + 20% novo
   - Mantém consistência entre frazes
   - Permite mudança gradual de tópico
```

### Exemplo

```
Frase 1: "o Brasil é um país grande"
  → tópico evolve: Brasil ⊕ país ⊕ grande

Frase 2: "a cultura brasileira é rica"
  → usa tópico anterior para manter coesão
  → "brasileiro" é similar a "Brasil" (mesma raiz)
```

---

## 3. Coesão Semântica

### Amostragem Coesa

```
score = α × coesão + (1-α) × frequência

coesão = base_hamming + suffix_bonus

base_hamming = sigmoid(10 × (similaridade - 0.5))
suffix_bonus = 0.2 se mesmo sufixo que última palavra
```

### Pipeline de Geração

```
1. CUBO hierárquico → prediz classe POS
2. T2 multi-hop → concordância head/avô
3. T3 cascata → candidatos morfológicos
4. Amostragem coesa → score(α*coesão + (1-α)*frequência)
5. SNN confirma classe (82%)
6. Atualiza tópico
```

### Backoff

```
CUBO → classe → T3 refina → coesão desempata → SNN confirma
```

---

## 4. SNN (Rede Neural Spiking)

### Arquitetura

```
Input (30 neurônios):
  Class(7) + Gender(2) + Number(2) + Tense(5) + Person(3) + Style(3) + SynFunc(8)

Hidden (263 neurônios):
  7× classe + 256× T3 keys

Output (7 neurônios):
  SUBST(0) VERBO(1) ADJ(2) ART(3) ADV(4) PREP/CONJ(5) PONT(6)

16 timesteps LIF (tau=5.0, threshold=1.0, refrac=2)
STDP: 10 épocas, lr=0.001
```

### Métricas

| Métrica | Valor |
|---------|-------|
| Top-1 accuracy | 82.0% |
| Top-3 accuracy | 85.6% |
| Exemplos treino | 1.055 |
| Épocas | 10 |

---

## 5. Corpus

### Fontes

| Fonte | Tokens | Arquivos |
|-------|--------|----------|
| Wikipédia PT | 359K | 57 artigos |
| Romances Gutenberg | 151K | 5 romances |
| **Total** | **510K** | 62 arquivos |

### Romances

| Texto | Autor | Tokens |
|-------|-------|--------|
| Iracema | José de Alencar | 27K |
| O Guarany | José de Alencar | 56K |
| A Pata da Gazela | José de Alencar | 34K |
| Quincas Borba | Machado de Assis | 75K |
| Yayá Garcia | Machado de Assis | 55K |

---

## 6. Métricas Finais

| Componente | Métrica | Valor |
|------------|---------|-------|
| GRAPH | Embeddings | 50.030 |
| GRAPH | Decomposição | 69.7% |
| T3 | Entradas | 1.055 |
| T2 | Padrões | 3.903 |
| SNN | Top-1 | 82.0% |
| SNN | Top-3 | 85.6% |
| Corpus | Tokens | 510.291 |
| Afixos | Componentes | 613 |

---

## Commits

```
68dfd8d — topic tracking + cohesive sampling
5b58bac — morphological vocabulary (174 components)
b5cd30b — expanded vocabulary (613 components)
7aa1059 — suffix bonus in generation
```

---

*AMADEUS v6.1 — Setembro 2026*
