package core;
import java.util.*;

/**
 * Implements a Uniform-Cost Search (UCS) algorithm.
 * <p>
 * This class extends {@link AbstractSearch} and provides the specific "fringe"
 * implementation required for UCS, which is a priority queue ordered by accumulated cost.
 *
 * @see Ilayout
 * @see AbstractSearch
 *
 * @author Brandon Mejia
 * @version 2025-09-27
 */
public class GSolver extends AbstractSearch {
    /**
     * Creates the fringe (the "abertos" list) for Uniform-Cost Search.
     *
     * @return A {@link PriorityQueue} that orders states by their accumulated cost (`g`).
     * A secondary comparison on the sequence ID is used for tie-breaking, ensuring
     * a stable, FIFO-like behavior for states with the same cost.
     */
    @Override
    protected Queue<State> createFringe() {
        // Break ties using sequence ID for consistent ordering
        return new PriorityQueue<>(
                Comparator.comparingDouble(State::getG).thenComparingLong(State::getSequenceId)
        );
    }
}
