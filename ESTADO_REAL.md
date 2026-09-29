# Estado Real do Amadeus — medido em 2026-09-28

Documento de diagnóstico. A `ARCHITECTURE.md` descreve o projeto como ele foi
desenhado; este descreve o que foi medido no código, com números.

Comandos usados para produzir cada número estão indicados.

---

## 1. Dois caminhos de execução

O projeto tem **duas implementações de inferência**, e elas não conversam entre
si. Isso é a confusão principal de quem chega ao repositório.

| Caminho | Binário | Estado |
|---------|---------|--------|
| Pipeline de gramática | `src/bin/benchmark_wikipedia.rs` | **Funciona.** Gera português |
| Transformer | `src/bin/amadeus_m.rs` | Esqueleto. Pesos aleatórios |

### Pipeline de gramática (o que funciona)

Carrega `training/wikipedia_grammar.bin` e gera texto:

```bash
cargo run --release --bin benchmark_wikipedia
```

```
▸ FASE 1: Carregando gramática Wikipédia...
  Gramática carregada: 1379 lex, 4968 T2, 99301 GRAPH
  SNN construída

  "o Brasil é um país"
    Cascade: o brasil é um país brasil
    SNN:     o brasil é um país
```

O `.bin` **não carrega o CUBO**, e é por isso que esse binário roda sem
consumo excessivo de memória.

### Transformer (esqueleto)

```bash
cargo run --release --bin amadeus_m
```

```
Generating 20 tokens...
//////////r[231]/r[231]/r[231]/r
```

A saída é um loop degenerado. Duas causas somadas:

1. `src/bin/amadeus_m.rs:27` chama `AmadeusMModel::new(cfg)`, que cria pesos
   **aleatórios**. O modelo treinado de 795 MB não é carregado.
2. `src/main.rs:50` faz `prompt.bytes()` como tokens, com o comentário
   `// In a real impl: let tokens = tokenizer.encode(&prompt)`.

Existe um loader pronto (`src/amadeus_m/dual_loader.rs`, com `read_tensor_f32`
e `list_tensors`) e um modelo treinado
(`training/wikipedia_grammar.gguf`), mas nada conecta os dois.

---

## 2. O modelo de 795 MB e seu custo de memória

`training/wikipedia_grammar.gguf` — 775 MB, 19 tensores, GGUF v3.

### Onde estão os bytes

```
cubo.clause.blob     721,5 MB   97,5%   <- 189.135.413 bytes
lexicon.forms          3,0 MB
t2.head.data           2,3 MB
t3.exact.data          2,2 MB
t3.css.data            2,2 MB
demais (14 tensores)   ~8 MB
```

### Onde está a RAM

```
arquivo lido:      739 MB
rss apos read:     741 MB      <- I/O e quantização não são o gargalo
load_gguf total:  3632 MB      <- +2,9 GB na deserialização
cubo clause table: 7670280    <- 7,67 milhões de contextos
```

Deserializar 7,67M entradas em `HashMap<Vec<u128>, HashMap<u32, f32>>` custa
~380 bytes por entrada. Não é quantização: `snn.synapses_ih` tem 7.890
elementos e ocupa 0,03 MB.

### Verificação: o round-trip do blob está correto

Vale registrar porque a hipótese natural ("o reader corrompe o blob") é falsa:

```
save:  numel = data.len()/4   -> shape=[189135413]
read:  byte_size = numel*4    -> 756541652 bytes = tamanho real do blob
n_contexts declarado:  7.670.280
contextos na tabela:   7.670.280
```

Simétrico, correto, sem leitura fora dos limites.

### Distribuição de frequência

```
contextos totais:  7.670.280
mediana:  total=1        <- metade dos contextos aparece UMA vez
p75:      total=1
p90:      total=1
p99:      total=3

poda total<2:  mantém   220.782 ( 2,9%) contextos, 15% da massa
poda total<3:  mantém    92.307 ( 1,2%) contextos, 12% da massa
```

90% dos contextos são únicas. Podar `total<2` remove 97% das entradas e 85%
da massa — não é remoção de ruído, é descarte do aprendido. Por isso a poda
foi descartada e o sharding foi escolhido.

---

## 3. Sharding (implementado, commit `91277ee`)

`src/amadeus_m/shard.rs`. Um arquivo `.cuboshard` por flush, índice
contexto→offset em RAM, registros lidos via mmap.

```
memória real  3,6 GB -> 1,0 GB
geração       4,0s  -> 3,9s   (1,02x)
saída         idêntica
```

A memória real foi confirmada com `malloc_trim` (2,9 GB → 1,0 GB) e com uma
alocação de 2 GB subsequente, que não aumentou o RSS.

**Ressalva:** o spill roda *depois* do pico de RAM. Não resolve OOM quando dois
processos carregam o modelo juntos (testes em paralelo). Resolver exigiria ler
o blob direto do mmap sem desserializar — não implementado.

---

## 4. Correção de não-determinismo na amostragem

Bug pré-existente, encontrado pelo teste de paridade do sharding.

`HyperCube::sample_lex_modulated` percorria `dist` (um `HashMap`) na ordem de
iteração para a varredura cumulativa da amostra. A ordem de iteração de um
`HashMap` depende do layout interno, que muda entre construção por
`entry().or_insert()` e por `with_capacity()`. Resultado: dois CUBos com o
mesmo conteúdo, montados de formas diferentes, geravam sequências distintas
com a mesma semente.

Corrigido ordenando por lexema em quatro pontos: `probs` na amostragem, as
chaves do ramo de exploração, o top-k da modulação GRAPH e o desempate do
`sort_by` (que é instável).

A geração agora é reprodutível entre builds de tabela distintos. Este é o
ganho de mais valor do trabalho, e é independente do sharding.

---

## 5. Como rodar cada coisa

### Binário de conversa / REPL

**Path: `src/bin/amadeus_m_train.rs`**

```bash
cargo run --release --bin amadeus_m_train          # REPL interativo
cargo run --release --bin amadeus_m_train --say="o gato"
cargo run --release --bin amadeus_m_train --train-morphology --iterations 5 --order 3
```

Comandos do REPL: `temp`, `order`, `explore`, `maxlen`, `status`, `/porque`,
`/ast`, `sair`.

> O `README.md` aponta estes comandos para `--bin amadeus_m`. Está errado: os
> flags e o REPL estão em `amadeus_m_train`. `amadeus_m` não aceita nenhum
> deles.

Saída real do REPL hoje:

```
─── Episódio 1 ───
  Você: /porque gato
  Amadeus: gato gato gato gato porque gato porque gato gato gato gato ...
```

Repetição forte. Causa: `amadeus.grammar7` tem 448 bytes e léxico de 17
entradas (é um checkpoint de teste, não o modelo de 7,67M contextos).

### Benchmark com o modelo real

```bash
cargo run --release --bin benchmark_wikipedia
```

### Inspeção do GGUF

```bash
# 5 testes de caracterização do formato
cargo test --test gguf_characterization --release

# 7 testes de paridade do shard
cargo test --test shard_parity --release

# 3 testes de paridade de geração RAM vs sharded
cargo test --test shard_generation --release

# carga do modelo real (lento, ~100s, precisa de ~4 GB)
cargo test --test grammar_load --release -- --test-threads=1
```

---

## 6. Lacunas conhecidas

| Lacuna | Impacto | Onde |
|--------|---------|------|
| `amadeus_m` não carrega o modelo | Binário transformer inútil | `src/bin/amadeus_m.rs:27` |
| `main.rs` não tokeniza prompt | Prompt ignorado | `src/main.rs:50` |
| `grammar7` é checkpoint de teste | REPL repete tokens | `amadeus.grammar7` (448 B) |
| README aponta binário errado | Comandos falham | `README.md:100-106` |
| Corpus Underworld vazio | Compilador sem raízes | `training/empty_underworld/lingua/raizes/` |
| 9 docs de arquitetura, 3 sobrepostos | Risco de ler o errado | `ARCHITECTURE_HYBRID.md` (jul), `PHILOSOPHICAL_STACK.md` (mai) |

Vigente: `ARCHITECTURE.md` (v6.3, 16/set). `ARCHITECTURE_SNN.md` (14/set) e
`AMADEUS.md` (14/set) são complementares. `ARCHITECTURE_HYBRID.md` (jul) e
`PHILOSOPHICAL_STACK.md` (mai) estão superados e não têm marcação de obsolescência.

---

## 7. Cobertura de testes

Antes desta sessão: 7 `#[test]` em 63 arquivos, concentrados em `self_play`,
`morph_vocab` e `rhetoric`. Nenhum verificava geração.

Adicionados 15:

| Arquivo | Testes | Verifica |
|---------|--------|----------|
| `gguf_characterization.rs` | 5 | Formato, tensores e dimensões do artefato real |
| `shard_parity.rs` | 7 | Conteúdo lido do shard == HashMap, cascade, corrompido |
| `shard_generation.rs` | 3 | Geração idêntica RAM vs sharded, auto-flush |

`grammar_load.rs` foi reescrito: o teste anterior carregava dois modelos de
795 MB simultâneos e era morto por OOM. Agora carrega um e gera a partir dele.
