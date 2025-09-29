package core;
import java.util.*;

/**
 * Implements the A* search algorithm for the array sorting problem.
 * This class extends {@link AbstractSearch} and provides the specific "fringe"
 * implementation required for A*, which is a priority queue ordered by the total estimated cost `f = g + h`.
 *
 * @see Ilayout
 * @see AbstractSearch
 *
 * @author Brandon Mejia
 * @version 2025-09-27
 */
public class GSolver extends AbstractSearch
{
    /**
     * Creates the fringe (the open set) for the A* search algorithm.
     * @return A {@link PriorityQueue} that orders states first by their total estimated cost (`f`),
     * and then by a sequence ID as a tie-breaker, ensuring FIFO behavior for equal-cost states.
     */
    @Override
    protected Queue<State> createFringe() 
    {
        // Uniform-Cost Search uses g (path cost) as the primary sorting criterion.
        // The comparator first orders by g, then by sequence ID for FIFO tie-breaking.
        return new PriorityQueue<>(
                Comparator.comparingDouble(State::getG)
                        .thenComparingLong(State::getSequenceNumber));
    }
}
