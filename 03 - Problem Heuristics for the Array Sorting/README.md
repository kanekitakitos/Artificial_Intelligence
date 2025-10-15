# Array Sorting with Heuristic Search

Este proyecto explora la resolución de un problema de ordenación de arrays utilizando algoritmos de búsqueda en espacios de estados. El objetivo es encontrar la secuencia de intercambios (swaps) de mínimo coste para transformar un array inicial en un array objetivo.

El coste de cada intercambio depende de la paridad de los números involucrados:
- **Par-Par**: Coste 2
- **Impar-Impar**: Coste 20
- **Par-Impar**: Coste 11

## Algoritmos Implementados

El proyecto implementa y compara dos algoritmos de búsqueda fundamentales:

1.  **`GSolver` (Búsqueda de Coste Uniforme - UCS):**
    - Es un algoritmo de búsqueda **no informada**.
    - Explora el espacio de estados expandiendo siempre el nodo con el menor coste acumulado (`g(n)`) desde el inicio.
    - Garantiza encontrar la solución de coste óptimo, pero puede ser extremadamente lento en problemas complejos, ya que explora muchas rutas innecesarias al no tener "sentido de la dirección".

2.  **`AStarSearch` (Búsqueda A*):**
    - Es un algoritmo de búsqueda **informada**, mucho más eficiente.
    - Utiliza una función de evaluación `f(n) = g(n) + h(n)`, donde:
        - `g(n)` es el coste real desde el estado inicial hasta el estado `n`.
        - `h(n)` es una **heurística** que estima el coste mínimo desde `n` hasta el estado objetivo.
    - La clave de su rendimiento reside en la calidad de la heurística `h(n)`.

---

## La Heurística de A*: El Corazón de la Eficiencia

La superioridad del algoritmo A* en este proyecto se debe a una heurística (`h(n)`) muy sofisticada y precisa, implementada en la clase interna `ArrayCfg.Heuristic`. Esta heurística proporciona una estimación muy ajustada del coste real restante, permitiendo al algoritmo podar ramas enteras del árbol de búsqueda y encontrar la solución óptima de forma increíblemente rápida.

La lógica se basa en el concepto matemático de la **descomposición de permutaciones en ciclos disjuntos**.

### 1. ¿Qué es la Descomposición en Ciclos?

El problema de ordenar el array se puede ver como transformar una permutación de números en otra. Cualquier permutación se puede descomponer en un conjunto de "ciclos" independientes.

**Ejemplo:**
- Array Actual: `[B, C, A]`
- Array Objetivo: `[A, B, C]`

Aquí, el elemento `B` está en la posición de `A`, `C` está en la posición de `B`, y `A` está en la posición de `C`. Esto forma un único **3-ciclo**: `A -> B -> C -> A`.

La idea fundamental es que **los ciclos son subproblemas independientes**. El coste total para ordenar el array es la suma de los costes para resolver cada ciclo por separado. Un ciclo de longitud `k` siempre requiere un mínimo de `k-1` intercambios para ser resuelto.

### 2. Estrategia Híbrida para Calcular el Coste de los Ciclos

La heurística no se conforma con una estimación simple. Utiliza una **estrategia híbrida** para calcular el coste de cada ciclo con la máxima precisión posible, dependiendo de su tamaño:

#### a) 2-Ciclos (Intercambios Simples)
- **Lógica:** Un ciclo de longitud 2 (ej: `A` en la posición de `B` y `B` en la de `A`) se resuelve con un único intercambio.
- **Cálculo:** La heurística calcula el **coste exacto** de ese único intercambio (`calculateCost(A, B)`). Es la estimación más precisa posible.

#### b) Ciclos Pequeños (3 a 5 elementos)
- **Lógica:** Para ciclos de tamaño `k` entre 3 y 5, la heurística realiza una **búsqueda por fuerza bruta** para encontrar el **coste óptimo real** para resolver ese ciclo.
- **Cálculo:** Explora todas las secuencias válidas de `k-1` intercambios entre los elementos del ciclo y se queda con la de menor coste. Aunque es computacionalmente intensivo, para un `k` tan pequeño el coste es trivial, pero la ganancia en precisión para la heurística es enorme.

#### c) Ciclos Grandes (> 5 elementos)
- **Lógica:** Para ciclos más grandes, la fuerza bruta sería demasiado lenta (explosión combinatoria). En su lugar, se utiliza un **algoritmo voraz (greedy)** rápido y admisible.
- **Cálculo:** Se agrupan todos los elementos de los ciclos grandes y se estima el coste total simulando repetidamente el tipo de intercambio más barato posible con los elementos disponibles (primero todos los par-par, luego los par-impar, y finalmente los impar-impar). Esto garantiza una estimación conservadora que nunca sobreestima el coste real.

### 3. Admisibilidad: La Garantía de Optimalidad

La heurística es **admisible**, lo que significa que **nunca sobreestima el coste real** para llegar al objetivo. Esto es crucial, ya que es la condición que garantiza que A* encontrará la solución óptima.

La admisibilidad se mantiene porque:
- Para 2-ciclos, usa el coste **exacto**.
- Para ciclos pequeños, encuentra el coste **óptimo**.
- Para ciclos grandes, usa una estimación voraz que representa el **mejor caso posible**.

Gracias a esta combinación de precisión y eficiencia, el algoritmo A* es capaz de resolver problemas muy complejos en milisegundos, mientras que un algoritmo no informado como `GSolver` tardaría minutos, horas o incluso más.

## Cómo Ejecutar el Proyecto

1.  **Clase Principal:** El punto de entrada es `Main.java`.
2.  **Entrada:** El programa espera dos líneas desde la entrada estándar:
    - La primera línea es el array inicial (números separados por espacios).
    - La segunda línea es el array objetivo.

    **Ejemplo de entrada:**
    ```
    2 4 6 8 10 12 1 3 5 7 9 11
    1 3 5 7 9 11 2 4 6 8 10 12
    ```
4.  **Salida:** El programa imprimirá un único número: el coste total mínimo de la solución.

## Estructura del Proyecto

- **`src/core`**: Contiene las clases principales del motor de búsqueda.
  - `AbstractSearch.java`: Clase base abstracta para los algoritmos de búsqueda.
  - `AStarSearch.java`: Implementación de A*.
  - `GSolver.java`: Implementación de Búsqueda de Coste Uniforme.
  - `ArrayCfg.java`: Representación del estado del problema y la lógica de la heurística.
  - `Ilayout.java`: Interfaz que define la estructura de un estado.
- **`src/test`**: Contiene las pruebas unitarias y de rendimiento.
  - `AStarSearchTest.java`: Pruebas para el algoritmo A*.
  - `GSolverTest.java`: Pruebas para el algoritmo `GSolver`.
  - `ComparisonTest.java`: Pruebas de benchmark que comparan la velocidad de ambos algoritmos en casos complejos.