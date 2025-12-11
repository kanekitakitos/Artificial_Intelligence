# Guide to Creating High-Quality Java (JavaDoc) Documentation

This guide defines the standard of excellence for documenting classes and methods in our project. The goal is to create documentation that is clear, complete, and accelerates the understanding and use of the code by other programmers. We will use the `AbstractSearch` class documentation as our "gold standard" example.

## General Structure of a JavaDoc Comment Block

Every JavaDoc block for a class should follow this structure:

1.  **Concise Summary (First Sentence):** The first sentence must be a short and direct description of the class's main purpose. It must end with a period.
    -   *Exemplo:* `An abstract base class for implementing state-space search algorithms using the Template Method pattern.`

2.  **Detailed Description (`<p>`):** After the summary, add one or more paragraphs that expand on the functionality. Explain *what* the class does, *why* it exists, and its main responsibilities. Mention any important design patterns (`{@link Template Method pattern}`) or crucial implementation details.
    -   *Exemplo:* `This class provides the core search loop, management of open (fringe) and closed lists, and path reconstruction...`

3.  **Implementation or Usage Guide (`<h3>`):** If the class is abstract, an interface, or complex to configure, include a section that teaches other programmers how to use it.
    -   Use o cabeçalho `<h3>Implementing a Search Strategy</h3>` ou `<h3>Example Usage</h3>`.
    -   Explain the extension points (e.g., abstract methods that need to be implemented).

4.  **Concrete Examples (`<h4>`, `<ul>`, `<pre>{@code}`):** This is the most critical part. Provide ready-to-use code examples for the most common usage scenarios.
    -   Use `<h4>` para cada cenário específico (ex: `Uniform-Cost Search (UCS)`).
    -   Use `<p>` to briefly describe the scenario.
    -   Use a `<ul>` list to detail the components of the solution.
    -   Use `<b>` para destacar termos-chave como **Data Structure** ou **Implementation**.
    -   Provide the exact code inside a `<pre>{@code ... }</pre>` block. The code must be complete, correct, and formatted.
    -   *Exemplo (para BFS):*
        ```java
        /**
         * <h4>Breadth-First Search (BFS)</h4>
         * <p>
         * Expands nodes level by level. Guarantees the shortest path in terms of number of steps,
         * assuming all step costs are equal.
         * </p>
         * <ul>
         *   <li><b>Data Structure:</b> A simple FIFO queue, like {@link java.util.LinkedList}.</li>
         *   <li><b>Implementation:</b>
         *       <pre>{@code return new LinkedList<>();}</pre>
         *   </li>
         * </ul>
         */
        ```

5.  **JavaDoc Block Tags (`@`):** End the documentation with the appropriate Javadoc tags, in the following order:
    -   `@preConditions` (Optional): What must be true *before* a key method is called or a subclass is used.
    -   `@postConditions` (Optional): What will be true *after* a key method is executed.
    -   `@see`: Links to other classes or methods that are crucial for understanding the context of this class. Use one per line.
    -   `@author`: Your name.
    -   `@version`: The date in `YYYY-MM-DD` format.

## Quality Checklist

When writing documentation, check the following points:

-   **[ ] Clarity and Conciseness:** Does the first sentence perfectly summarize the class?
-   **[ ] Purpose (The "Why"):** Does the documentation explain why the class is necessary and what problem it solves?
-   **[ ] Practical Guide:** If the class needs to be extended or configured, is there an `<h3>` section that explains how?
-   **[ ] Code Examples:** Are there code examples (`<pre>{@code...}`) for the most common use cases? Are the examples "copy-and-paste" friendly?
-   **[ ] HTML Formatting:** Does the documentation use `p`, `h3`, `h4`, `ul`, `li`, and `b` to structure the information legibly?
-   **[ ] Internal Links (`@link`):** Are references to other classes, methods, or design patterns in the text correctly linked with `{@link ...}`?
-   **[ ] Complete JavaDoc Tags:** Are the `@see`, `@author`, and `@version` tags present and correct?
-   **[ ] Professional Language:** Is the text written in English (or the project's default language) in a clear and professional manner?

---

By rigorously following this guide, we ensure that our code's documentation will be as useful and well-constructed as the code itself.