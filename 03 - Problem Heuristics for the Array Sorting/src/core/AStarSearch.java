package core;

import java.util.Comparator;
import java.util.PriorityQueue;
import java.util.Queue;

/**
 * Implements the A* search algorithm.
 * <p>
 * This class extends {@link AbstractSearch} and provides the specific "fringe"
 * implementation required for A*, which is a priority queue ordered by the
 * evaluation function f(n) = g(n) + h(n).
 *
 * @author Brandon Mejia
 * @version 2025-10-14
 */
public class AStarSearch extends AbstractSearch {
    /**
     * Creates the fringe for A* Search.
     *
     * @return A {@link PriorityQueue} that orders states by f(n) = g(n) + h(n).
     * Tie-breaking is done using the sequence ID.
     */
    @Override
    protected Queue<State> createFringe() {
        return new PriorityQueue<>(
                Comparator.comparingDouble((State s) -> s.getG() + s.getH()).thenComparingLong(State::getSequenceId));
    }
}