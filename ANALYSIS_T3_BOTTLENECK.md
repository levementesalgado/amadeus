# Análise: Gargalo do T3 (485 entradas)

> Investigação sobre por que a tabela T3 do TripleGrammar está em 485 entradas
> mesmo com 946K tokens de treino e 10K+ entradas no léxico.

---

## A Chave do T3

T3 mapeia `(morph: u16, syn_func: u8, style: u16)` → distribuição sobre `lex: u32`.

Espaço teórico: 2¹⁶ × 2⁸ × 2¹⁶ ≈ 134M combinações possíveis.

## A Descoberta: `style = 0` em Todo o Treino

Nenhuma fonte de dados de treino popula `style`:

| Fonte | Arquivo | `style` |
|-------|---------|---------|
| `Compiler::compile()` | `compiler.rs:124-140` | `0` (default de `Token7::new()`) |
| PCFG `mk_tok()` | `pcfg.rs:41-45` | `0` (default de `Token7::new()`) |
| `assign_dependencies()` | `syntax.rs:7-34` | Nunca toca em `style` |
| `generate()` / `sample_token()` | `triple_grammar.rs:221` | Seta `style` durante **geração**, mas não durante treino |

**Resultado:** Toda chave T3 efetivamente é `(morph, syn_func, 0)` — style não contribui variação.

## Por Que 485?

Com `style = 0`, a chave se reduz a `(morph, syn_func)`.

### Distribuição de `morph`

`morph` codifica (bits 0-10):
| Campo | Bits | Valores |
|-------|------|---------|
| class | 0-2 | 7 (S, V, ADJ, ART, PREP, PONT, OUT) |
| gender | 3 | 2 (M, F) |
| number | 4 | 2 (S, P) |
| tense | 5-7 | 5 (P, IMP, FUT, PRET, INF) |
| person | 8-10 | 3 (1, 2, 3) |

Teoricamente: 7 × 2 × 2 × 5 × 3 = 420 combinações.
Na prática (10K+ entradas no léxico): ~300 valores distintos de `morph`.

### Distribuição de `syn_func`

| Classe | `syn_func` possíveis | Count |
|--------|---------------------|-------|
| S (0) | SUJEITO(0), OD(1), OI(2), ROOT(4) | 4 |
| V (1) | ROOT(4), PRED(7) | 2 |
| ADJ (2) | ADN(5), PRED(7) | 2 |
| ART (3) | ADN(5), ROOT(4) | 2 |
| PREP (4) | PREP(6) | 1 |
| PONT (5) | 0 | 1 |
| OUT (6) | ADV(3), ROOT(4) | 2 |
| **Total teórico** | | **14** (13 observados, falta ADJ+ROOT) |

### Produto Esperado

~300 morph × ~1.6 syn_func/morph = **~480 pares únicos**

Observado: **485**. Casamento quase perfeito.

## Por Que CSS=13, CS=13, CST=7?

Com `style=0`:

| Tabela | Chave | Efetiva | Entradas | Esperado |
|--------|-------|---------|----------|----------|
| T3 | (morph, syn_func, style) | (morph, syn_func) | **485** | ~480 |
| CSS | (class, syn_func, style) | (class, syn_func) | **13** | 14 (13 obs.) |
| CS | (class, syn_func) | (class, syn_func) | **13** | 14 (13 obs.) |
| CST | (class, style) | (class) | **7** | 7 |

CSS == CS em número porque style=0; CST == Class pelo mesmo motivo.

## Conclusão

485 entradas **não é um bug** — é o ponto de saturação natural dado que:

1. `style` nunca é populado durante o treino (sempre 0)
2. O espaço de variação morfossintática do corpus é de ~480 combinações (morph × syn_func)
3. Com 946K tokens, o modelo já viu todas as combinações possíveis

### A Cascata Funciona Corretamente

```
T3 (morph, syn_func) → 485 → cobre 100% das combinações observadas
  ↓ fallback
CS (class, syn_func) → 13 → 13 de 14 teóricas
  ↓ fallback
Class → 7 → 100% das classes
```

O CUBO (1.24M entradas, 8D) é o preditor principal; T3 é refinamento/confirmação.

## Para Aumentar a Cobertura T3

A alavanca mais impactante: **popular `style` durante o treino**.

| Abordagem | Esforço | Impacto |
|-----------|---------|---------|
| Anotar registro (formal/informal/neutro) por lexema no compilador | Médio | Multiplica T3 ×2-3× |
| PCFG com `style` aleatório | Baixo | Popula T3 para PCFG |
| Detectar gênero textual dos `.md` | Alto | Estilo realista |

---

*Investigado em Junho 2026 — AMADEUS v3 (CUBO 8D, EM λ, Underworld 946K)*
