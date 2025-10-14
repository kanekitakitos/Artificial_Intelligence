package core;

import java.util.*;

/**
 * Implements a generic Best-First search algorithm by extending the {@link AbstractSearch} template.
 * <p>
 * "Best-First" is a family of search algorithms that explore a graph by expanding the most
 * promising node chosen according to a specified rule. This class extends {@link AbstractSearch}
 * and implements the search strategy by providing a specific fringe data structure.
 *
 * <h3>Search Strategy: Uniform-Cost Search (UCS)</h3>
 * <p>
 * This implementation uses a {@link PriorityQueue} ordered solely by the accumulated path cost (`g(n)`).
 * This makes it functionally equivalent to a <strong>Uniform-Cost Search (UCS)</strong>, which guarantees
 * finding the least-cost path from the start to the goal.
 * </p>
 *
 * <h4>Adaptation to A* Search</h4>
 * <p>
 * To adapt this class to an A* search, the comparator in {@link #createFringe()} would need to be
 * modified to include a heuristic function `h(n)`. For example:
 * </p>
 * <pre>{@code
 * Comparator<State> comparator = Comparator.comparingDouble((State s) -> s.getG() + s.getH());
 * return new PriorityQueue<>(comparator);
 * }</pre>
 *
 * @see AbstractSearch
 * @see Board
 * @see GSolver
 *
 * @author Brandon Mejia
 * @version 2025-09-30
 */
public class BestFirst extends AbstractSearch
{
    /**
     * Creates the fringe (the "abertos" list) for this search strategy.
     * <p>
     * For this implementation (equivalent to Uniform-Cost Search), it returns a
     * {@link PriorityQueue} that orders states by their accumulated path cost (`g`).
     * A secondary comparison on the sequence ID is used for tie-breaking, ensuring
     * a stable, FIFO-like behavior for states with the same cost.
     *
     * @return A {@link PriorityQueue} configured for Uniform-Cost Search.
     */
    @Override
    protected Queue<State> createFringe()
    {
        // Using a PriorityQueue ordered by G makes this a Uniform-Cost Search.
        // The sequence ID is used as a tie-breaker for stability.
        return new PriorityQueue<>(
                Comparator.comparingDouble(State::getG).thenComparingLong(State::getSequenceId)
        );
    }
}