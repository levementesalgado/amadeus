```
                      ╔═╗╔╗╔╔═╗╔═╗╔═╗╔╦╗╔═╗╔╗╔
                      ║ ║║║║║ ║║ ║╚═╗ ║║║║║║║║
                      ╚═╝╝╚╝╚═╝╚═╝╚═╝═╩╝╚═╝╝╚╝
            CPU-first LLM inference engine • Mizu OS Project
```

Amadeus é um motor de inferência **100% Rust**, CPU-only, que carrega modelos GGUF reais **e** implementa uma gramática tabular experimental sobre sequências morfológicas — o **AMADEUS-M**.

---

## Duas Almas

### Amadeus (clássico) — `cargo run --release -- modelo.gguf`

Carrega modelos GGUF (Ollama, llama.cpp) com quantização e transformer decoder padrão.

### AMADEUS-M (morfológico) — `cargo run --release --bin amadeus_m`

Opera exclusivamente sobre números. Nenhum texto entra no modelo. O **Compilador Morfológico** traduz texto ↔ `Token7` (8 campos inteiros) e o modelo aprende padrões nessas sequências numéricas. Não é uma rede neural. Não usa retropropagação.

```
texto → compilador → Vec<Token7> → assign_dependencies() →
  ├─ GRAPH: random indexing (u32 semântico)
  ├─ CUBO 7D: contexto multidimensional → prediz próximo lex
  ├─ T2 multi-hop: head + avô → MORPH + STYLE
  └─ T3 cascata: 5 níveis de fallback → confirma lexical
```

---

## Amadeus-M: Arquitetura

### Compilador Morfológico (`compiler.rs`)

Carrega léxico estruturado de `underworld/lingua/raizes/*.txt`. Cada palavra portuguesa mapeia para um `Token7` com 8 campos:

| Campo | Tipo | Bits | Função |
|-------|------|------|--------|
| `lex` | u32 | 32 | ID da raiz lexical |
| `morph` | u16 | 16 | Classe(3) + gênero(1) + número(1) + tempo(3) + pessoa(3) |
| `syn_off` | i16 | 16 | Deslocamento negativo para o head (0=root) |
| `syn_func` | u8 | 8 | Função sintática: sujeito(0), OD(1), OI(2), ADV(3), root(4), ADN(5), PREP(6), pred(7) |
| `orth` | u8 | 8 | Ortografia |
| `punct` | u8 | 8 | Pontuação associada |
| `style` | u16 | 16 | Estilo/afeto (modulado pelo Collatz) |
| `graph` | u32 | 32 | Embedding semântico (random indexing) |

### Grammar Tabular de 3 Níveis (`TripleGrammar`)

| Tabela | Entrada | Saída | O que aprende |
|--------|---------|-------|---------------|
| **CUBO 7D** | `histórico[LEX_HASH\|MORPH\|SYN_BIN\|SYN_FUNC\|PUNCT\|STYLE\|CLAUSE]` | `lex_t` | Contexto multidimensional → próxima palavra |
| **T2** | `(head.morph, head.style, curr_class)` | `curr_morph, curr_style` | Propagação de concordância (head imediato) |
| **T2+** | `(avo.morph, avo.style, curr_class)` | `curr_morph, curr_style` | Propagação multi-hop via avô (0.4 avô + 0.6 head) |
| **T3** | `(morph, syn_func, style)` → `(class, style)` → `class` | `lex_t` | Seleção lexical com 5 níveis de fallback |

Todas as tabelas são `HashMap` de contagens — sem redes neurais, sem backpropagation.

### GRAPH Embedding

Cada token recebe um `u32` via random indexing: XOR com rotação à direita dos vizinhos dentro de uma janela de 3. Similaridade semântica = `1 - Hamming(graph_a ^ graph_b) / 32`. Durante a amostragem no CUBO, o top-20 da distribuição tem massa espalhada entre pares com similaridade > 0.5 (alpha=0.15).

### Componentes

| Módulo | Arquivo | Função |
|--------|---------|--------|
| `Compiler` | `compiler.rs` | Mapeia texto ↔ Token7 + sintaxe + graph |
| `SyntaxAnalyzer` | `syntax.rs` | Atribui função sintática (assign_dependencies) |
| `HyperCube` | `hypercube.rs` | 7D context packing + interpolação Jelinek-Mercer |
| `TripleGrammar` | `triple_grammar.rs` | CUBO + T2 multi-hop + T3 cascata + clause stack |
| `SpikingNetwork` | `snn.rs` | Rede neural spiking (T3 alternativo, 16 timesteps LIF) |
| `EmbeddingCompressor` | `embedding_compressor.rs` | LLM embeddings (4096D) → GRAPH 32-bit |
| `DualGgufLoader` | `dual_loader.rs` | mmap dual-GGUF (grammar + modelo comercial) |
| `ConsciousnessState` | `consciousness.rs` | Soma, Foco, Collatz, Temperamento |
| `CollatzOscillator` | `collatz.rs` | Oscilador Collatz (seed=27) |
| `PedagogicalLoop` | `pedagogy.rs` | Ciclo: treinar → gerar → reforçar + /porque |
| `Teacher` | `teacher.rs` | Feedback afetivo como reforço |
| `Agnes` | `agnes.rs` | LLM pedagoga via API |
| `Constitution` | `constitution.rs` | 3 regras + 2 emendas mutáveis |
| `Friction` | `friction.rs` | Análise de fricção cognitiva |
| `SyntaxAST` | `syntax_ast.rs` | AST builder + clause stack |

### Treinamento

```
                  ┌─ PCFG (~9K tokens/iteração, concordância forçada)
Treinamento ──────┼─ Underworld (~73K tokens de 208 arquivos .md)
                  └─ Compilador morfológico (10.114 entradas)
```

**NENHUM TEXTO entra no modelo.** A gramática só vê `Vec<Token7>`. O treino roda em ~30s CPU.

### Como usar

```bash
# Compilação otimizada
RUSTFLAGS="-C target-cpu=native" cargo build --release

# Treino completo + interativo
cargo run --release --bin amadeus_m -- --train-morphology --iterations 5 --order 3

# Modo interativo
cargo run --release --bin amadeus_m

# Comandos interativos: temp, order, explore, maxlen, status, /porque, /ast, sair

# Auto-episódio
cargo run --release --bin amadeus_m -- --say="o gato"

# Agnes (LLM pedagoga)
cargo run --release --bin agnes_train
cargo run --release --bin agnes_train --auto --train-morphology
```

### Estado atual (v5.1 — léxico 10K+)

```
CUBO:  174.920 contextos de 7 dimensões
T2:        808 padrões de concordância (head)
T2+:       377 padrões (multi-hop, avô)
T3:        485 entradas lexicais
GRAPH:   1.682 embeddings u32
Compiler: 10.114 entradas no léxico
Underworld: 74.586 tokens (223+ arquivos)
Inferência: ~400 tok/s (CPU single-core)
Arquivo: ~7MB (grammar) + 66KB (compiler)
```

### SNN — Rede Neural Spiking

Substituto bio-inspirado da cascata T3. 30 input → 263 hidden → N output, 16 timesteps LIF.

```
Input: morph(7) + gender(2) + number(2) + tense(5) + person(3) + style(3) + syn_func(8) = 30
Hidden: 7×classe + 256×T3 keys = 263
Output: 1 neurônio por lex_id
Scoring: potencial membrana acumulado + spikes × 2
```

Detalhes em `ARCHITECTURE_SNN.md`.

### Pipeline Híbrido (LLM + Amadeus)

Mapear embeddings de modelos comerciais (Llama, Qwen, Phi) para o GRAPH 32-bit:

```
LLM GGUF (4096D F16) → EmbeddingCompressor → 32-bit GRAPH
                         ↓
                   interpolate_graphs(legacy, llm, 0.6)
                         ↓
                   HyperCube modulation (Hamming spread)
```

Detalhes em `ARCHITECTURE_HYBRID.md`.

### Oscilador Collatz

```
Sequência: se n é par → n/2 (colapso)
           se n é ímpar → 3n+1 (expansão)
```

| n | Modo | Exploração | Temperatura | Comportamento |
|---|------|-----------|-------------|---------------|
| **Par** | **Originalista** | reduzida (~0.5×) | baixa (~0.8×) | Conservador, segue o aprendido |
| **Ímpar** | **Vanguardista** | aumentada (~2.5×) | alta (~2.2×) | Criativo, explora novas combinações |

Seed padrão: 27. A cada token gerado, Collatz avança um passo.

### Comando `/porque`

Após uma resposta, digite `/porque` para ver o rastro de decisão token por token:

```
─── /porque — Rastro de Decisão ───
  [ 0] n=  41 VANG T=1.34 E=0.10 → consciencios (pos=2, bits=0x100090)
  [ 1] n= 124 ORIG T=1.34 E=0.03 → fogas (pos=2, bits=0x1000A0)
  ...
  Resumo: 18 Originalista / 14 Vanguardista / 32 passos
```

### Estrutura de diretórios

```
underworld/
  lingua/
    raizes/           # 9 CSV: substantivos, verbos, adjetivos, advérbios...
    morfologia/       # flexoes.txt, sufixos.txt, composicao.txt
    sintaxe/          # termos.txt, acessorios.txt, oracoes.txt, relacoes.txt
  arte/               # poesia, música, pintura, historias
  dialogos/           # conhecimento, poéticos, filosofia...
  conhecimento/       # história, energia, astronomia, matemática...
  tecnologia/         # computadores, dados, IA...
  filosofia/          # ser, tempo, sentido, ética...
  pessoas/            # família, trabalho, culturas, lugares...
  livro/              # volume_I..X (gerado por gerar_livro.py)
  gerado/             # conhecimento + tecnologia (gerado por expandir_lexico.py)
```

### Dependências

| Crate | Uso |
|-------|-----|
| `fastrand` | RNG |
| `unicode-segmentation` | Tokenizer |

### Licença

MIT — faz o que quiser.

```
     🦀  +  🧮  +  ☕  =  🤖  (só números, zero texto)
```

Feito com carinho como parte do ecossistema Mizu OS.
