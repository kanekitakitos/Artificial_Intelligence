package core;

import java.util.*;

/**
 * Implements a generic Best-First search algorithm by extending the {@link AbstractSearch} template.
 * In its current form, it behaves as a Uniform-Cost Search because it uses a priority queue
 * ordered by the accumulated path cost (`g`). It can be adapted for other "best-first"
 * strategies (like A*) by changing the comparator.
 *
 * @preConditions
 *                 - The `solve` method (inherited from `AbstractSearch`) must be called with valid layouts.
 * @postConditions
 *                  - A solution path iterator is returned, representing the lowest-cost path found.
 *
 * @see AbstractSearch
 *
 * @author Brandon Mejia
 * @version 2025-09-27
 */
public class BestFirst extends AbstractSearch
{
    /**
     * Creates the fringe for the Best-First search.
     * @return A {@link PriorityQueue} that orders states by their accumulated cost (`g`).
     */
    @Override
    protected Queue<State> createFringe()
    {
        // BestFirst, as currently implemented, uses Uniform Cost Search logic.
        return new PriorityQueue<>(Comparator.comparingDouble(State::getG));
    }
}