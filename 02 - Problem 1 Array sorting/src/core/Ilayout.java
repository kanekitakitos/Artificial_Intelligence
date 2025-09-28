package core;

import java.util.List;
/**
 * Defines the contract for a state (or "layout") within a state-space search problem.
 * Any class representing a problem state must implement this interface to be compatible
 * with the search algorithms provided in this project, such as {@link GSolver}.
 *
 * @author Brandon Mejia
 * @version 2025-09-27
 */
public interface Ilayout
{
    /**
     * Generates all valid successor states (children) reachable from the current state
     * in a single step.
     * @return A list of {@link Ilayout} objects representing the children.
     */
    List<Ilayout> children();

    /**
     * Checks if the current state is the goal state by comparing it to a given layout.
     * @param l The goal layout to compare against.
     * @return `true` if the current layout is the goal, `false` otherwise.
     */
    boolean isGoal(Ilayout l);


    /**
     * Gets the cost of the single step (or move) that led from a parent state to this one.
     * This is not the accumulated cost from the start, but the cost of the last action.
     * @return The cost from the parent state to this state.
     */
    double getK();

    /**
     * Provides a string representation of the layout.
     * This method is crucial for displaying the state in the output and for debugging purposes.
     * Classes implementing this interface must provide a meaningful `toString()` implementation.
     * @return A string representation of this layout.
     */
    @Override
    String toString();
}
