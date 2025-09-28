package core;
import java.util.*;

/**
 * Implements the Uniform-Cost Search (UCS) algorithm for the array sorting problem.
 * This class extends {@link AbstractSearch} and provides the specific "fringe"
 * implementation required for UCS, which is a priority queue ordered by accumulated cost.
 * This is the main solver class used by the `Main` application.
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
     * Creates the fringe (the open set) for the Uniform-Cost Search algorithm.
     * @return A {@link PriorityQueue} that orders states by their accumulated cost (`g`),
     * ensuring the lowest-cost state is always expanded next.
     */
    @Override
    protected Queue<State> createFringe() 
    {
        // GSolver implements Uniform Cost Search.
        return new PriorityQueue<>(Comparator.comparingDouble(State::getG));
    }
}
