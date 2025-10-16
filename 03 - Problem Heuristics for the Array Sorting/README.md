# Compêndio de Exemplos para Heurística A* Baseada em Ciclos

**Autor:** Brandon Mejia

Este documento serve como um guia detalhado e um compêndio de exemplos para uma heurística do algoritmo A*, focada na resolução de problemas de ordenação através da análise de ciclos de permutação.

## Índice

- [A Lógica Central da Heurística](#a-lógica-central-da-heurística)
  - [Fase 1: O Diagnóstico (A Pergunta Chave)](#fase-1-o-diagnóstico-a-pergunta-chave)
  - [Fase 2: O Mapeamento (A Resposta à Pergunta)](#fase-2-o-mapeamento-a-resposta-à-pergunta)
  - [Fase 3: A Estratégia (O Tratamento segundo o Diagnóstico)](#fase-3-a-estratégia-o-tratamento-segundo-o-diagnóstico)
- [Exemplos para Ciclos de Tamanho 2 (k=2)](#exemplos-para-ciclos-de-tamanho-2-k2)
  - [Exemplo 2.1: Troca Par-Par](#exemplo-21-troca-par-par)
  - [Exemplo 2.2: Troca Ímpar-Ímpar](#exemplo-22-troca-ímpar-ímpar)
  - [Exemplo 2.3: Troca Par-Ímpar](#exemplo-23-troca-par-ímpar)
- [Exemplos para Ciclos Pequenos (k=3, k=4)](#exemplos-para-ciclos-pequenos-k3-k4)
  - [Exemplo 3.1: Ciclo de Tamanho 3 (Misto)](#exemplo-31-ciclo-de-tamanho-3-misto)
  - [Exemplo 3.2: Ciclo de Tamanho 3 (Apenas Ímpares)](#exemplo-32-ciclo-de-tamanho-3-apenas-ímpares)
  - [Exemplo 3.3: Ciclo de Tamanho 4 (Misto)](#exemplo-33-ciclo-de-tamanho-4-misto)
- [Exemplos para Ciclos Grandes (k > 4)](#exemplos-para-ciclos-grandes-k--4)
  - [Caso A: Ciclos com Números Pares](#caso-a-ciclos-com-números-pares)
  - [Caso B: Ciclos Apenas com Números Ímpares](#caso-b-ciclos-apenas-com-números-ímpares)

---

## A Lógica Central da Heurística

Esta secção resume o fluxo de "pensamento" do algoritmo, desde a identificação do problema até à aplicação de uma estratégia específica.

### Fase 1: O Diagnóstico (A Pergunta Chave)

A primeira e mais importante pergunta que a heurística procura responder é:

> *"Como é que as peças que estão fora do lugar se ligam umas às outras?"*

### Fase 2: O Mapeamento (A Resposta à Pergunta)

A resposta a essa pergunta é encontrada ao construir a **estrutura de ciclos**. Uma vez completada esta fase, o algoritmo já não sabe apenas *quantas* peças estão erradas, mas sim *como* os seus erros se relacionam entre si.

### Fase 3: A Estratégia (O Tratamento segundo o Diagnóstico)

O algoritmo analisa cada ciclo de forma individual e escolhe uma estratégia segundo o **TAMANHO** desse ciclo.

1.  **Ciclo de Tamanho 2:** Resolve-se com um único intercâmbio direto. É o caso mais simples e o custo calculado é o custo real e exato.

2.  **Ciclo de Tamanho 3 ou 4:** Aplica-se a "força bruta" combinatória para testar todas as sequências de trocas e encontrar o custo ótimo real para resolver esse pequeno ciclo.

3.  **Ciclo de Tamanho Grande (k > 4):** Adota-se uma estratégia "inteligente e económica":

    *   **A) Se o ciclo contém pelo menos um número par:** Usa-se a estratégia do "pivô par" para evitar os dispendiosos intercâmbios entre ímpares. O custo é estimado com a fórmula de limite inferior (lower bound):
        `Custo = (Nº_pares - 1) * 2 + (Nº_ímpares) * 11`

    *   **B) Se o ciclo só contém números ímpares:** Faz-se uma análise de custo-benefício, comparando duas opções:
        1.  *Opção Interna:* Resolver o ciclo com `k-1` trocas internas entre ímpares (custo alto: `(k-1) * 20`).
        2.  *Opção "Empréstimo":* "Tomar emprestado" um número par de outra parte do array e usá-lo como pivô (custo: `k * 11`).

        A heurística escolhe o custo **MÍNIMO** entre estas duas opções.

---

## Exemplos para Ciclos de Tamanho 2 (k=2)

### Exemplo 2.1: Troca Par-Par

```
data: [1, 4, 3, 2]
goal: [1, 2, 3, 4]
```

-   **Identificação:** O ciclo é (índice 1 ↔ 3), envolvendo os valores `{4, 2}`.
-   **Análise e Custo:** Ambos os valores são Pares. A troca é Par-Par.
-   **Custo da Heurística = 2**

### Exemplo 2.2: Troca Ímpar-Ímpar

```
data: [5, 2, 3, 4, 1]
goal: [1, 2, 3, 4, 5]
```

-   **Identificação:** O ciclo é (índice 0 ↔ 4), envolvendo os valores `{5, 1}`.
-   **Análise e Custo:** Ambos os valores são Ímpares. A troca é Ímpar-Ímpar.
-   **Custo da Heurística = 20**

### Exemplo 2.3: Troca Par-Ímpar

```
data: [1, 2, 8, 4, 3]
goal: [1, 2, 3, 4, 8]
```

-   **Identificação:** O ciclo é (índice 2 ↔ 4), envolvendo os valores `{8, 3}`.
-   **Análise e Custo:** Um valor é Par (8) e o outro Ímpar (3). A troca é Par-Ímpar.
-   **Custo da Heurística = 11**

---

## Exemplos para Ciclos Pequenos (k=3, k=4)

### Exemplo 3.1: Ciclo de Tamanho 3 (Misto)

```
data: [2, 3, 1]
goal: [1, 2, 3]
```

-   **Identificação:** O ciclo é (0 → 1 → 2 → 0), envolvendo `{2, 3, 1}`.
-   **Análise e Custo:** A força bruta testa as sequências de 2 trocas. A sequência `(2,1)` e depois `(2,3)` custa `11+11=22`. A sequência `(3,1)` e depois `(2,3)` custa `20+11=31`. A heurística encontra o mínimo.
-   **Custo da Heurística = 22**

### Exemplo 3.2: Ciclo de Tamanho 3 (Apenas Ímpares)

```
data: [3, 5, 1]
goal: [1, 3, 5]
```

-   **Identificação:** O ciclo é (0 → 1 → 2 → 0), envolvendo `{3, 5, 1}`.
-   **Análise e Custo:** Todas as 2 trocas necessárias serão Ímpar-Ímpar. Custo = `2 * 20 = 40`.
-   **Custo da Heurística = 40**

### Exemplo 3.3: Ciclo de Tamanho 4 (Misto)

```
data: [2, 3, 4, 1]
goal: [1, 2, 3, 4]
```

-   **Identificação:** O ciclo é (0 → 3 → 2 → 1 → 0), envolvendo `{2, 3, 4, 1}`.
-   **Análise e Custo:** A força bruta testa sequências de 3 trocas. Uma sequência de baixo custo é `(2,1)` (custo 11), `(2,3)` (custo 11), `(2,4)` (custo 2), para um total de 24.
-   **Custo da Heurística = 24**

---

## Exemplos para Ciclos Grandes (k > 4)

### Caso A: Ciclos com Números Pares

#### Exemplo 4.1: Ciclo k=5 (2 Pares, 3 Ímpares)

```
data: [12, 13, 14, 15, 11]
goal: [11, 12, 13, 14, 15]
```

-   **Identificação:** Ciclo de tamanho 5 envolvendo `{12, 13, 14, 15, 11}`.
-   **Análise:** `N_pares=2` (12, 14), `N_ímpares=3` (11, 13, 15).
-   **Custo:** `(2-1)*2 + 3*11 = 1*2 + 33 = 35`.
-   **Custo da Heurística = 35**

#### Exemplo 4.2: Ciclo k=6 (4 Pares, 2 Ímpares)

```
data: [2, 4, 3, 6, 5, 8]
goal: [8, 2, 3, 4, 5, 6]
```

-   **Identificação:** Ciclo de tamanho 6 envolvendo `{2, 4, 3, 6, 5, 8}`.
-   **Análise:** `N_pares=4` (2, 4, 6, 8), `N_ímpares=2` (3, 5).
-   **Custo:** `(4-1)*2 + 2*11 = 3*2 + 22 = 6+22=28`.
-   **Custo da Heurística = 28**

#### Exemplo 4.3: Ciclo k=7 (1 Par, 6 Ímpares)

```
data: [2, 3, 5, 7, 9, 11, 1]
goal: [1, 2, 3, 5, 7, 9, 11]
```

-   **Identificação:** Ciclo de tamanho 7 envolvendo `{2, 3, 5, 7, 9, 11, 1}`.
-   **Análise:** Valores: `{2, 3, 5, 7, 9, 11, 1}`. `N_pares=1` (só o 2), `N_ímpares=6`.
-   **Custo:** `(1-1)*2 + 6*11 = 0*2 + 66 = 66`.
-   **Custo da Heurística = 66**

### Caso B: Ciclos Apenas com Números Ímpares

#### Exemplo 4.4: k=5 com "Empréstimo" Possível

```
data: [3, 5, 7, 9, 1, 100]
goal: [1, 3, 5, 7, 9, 100]
```

-   **Identificação:** Ciclo de tamanho 5 com `{3, 5, 7, 9, 1}`. O array contém um par (100) que não pertence ao ciclo.
-   **Análise e Custo:**
    -   Opção Interna: `(5-1) * 20 = 80`.
    -   Opção "Empréstimo": `5 * 11 = 55`.
-   **Custo da Heurística = min(80, 55) = 55**

#### Exemplo 4.5: k=6 SEM Pares no Array

```
data: [3, 5, 7, 9, 11, 1]
goal: [1, 3, 5, 7, 9, 11]
```

-   **Identificação:** Ciclo de tamanho 6 com `{3, 5, 7, 9, 11, 1}`. O array não tem pares.
-   **Análise e Custo:** A opção de "empréstimo" é impossível. Apenas a estratégia interna é válida. Custo = `(6-1) * 20 = 100`.
-   **Custo da Heurística = 100**

#### Exemplo 4.6: k=7 com "Empréstimo" Possível

```
data: [3, 5, 7, 9, 11, 13, 1, 2]
goal: [1, 3, 5, 7, 9, 11, 13, 2]
```

-   **Identificação:** Ciclo de tamanho 7 com `{3, 5, 7, 9, 11, 13, 1}`. O array contém um par (2) que não pertence ao ciclo.
-   **Análise e Custo:**
    -   Opção Interna: `(7-1) * 20 = 120`.
    -   Opção "Empréstimo": `7 * 11 = 77`.
-   **Custo da Heurística = min(120, 77) = 77**