package core;

import java.util.List;
/**
 * Defines the contract for a state (or "layout") within a state-space search problem.
 * <p>
 * This interface is a core component of the search framework, abstracting the problem-specific
 * details of a state from the generic search logic in {@link AbstractSearch}. Any class
 * representing a configuration of a problem (like a board in the 8-puzzle or an array
 * in a sorting problem) must implement this interface.
 *
 * <h3>Interface Responsibilities</h3>
 * <p>
 * An implementing class is responsible for defining:
 * </p>
 * <ul>
 *   <li><b>State Representation:</b> The internal data that defines the state.</li>
 *   <li><b>Successor Function:</b> The logic for generating all possible next states from the current one ({@link #children()}).</li>
 *   <li><b>Goal Test:</b> The condition to check if a state is the desired final state ({@link #isGoal(Ilayout)}).</li>
 *   <li><b>Step Cost:</b> The cost of a single move or transition ({@link #getK()}).</li>
 *   <li><b>Equality and Hashing:</b> For a correct behavior in data structures like {@link java.util.HashMap},
 *       implementations MUST override {@code equals(Object)} and {@code hashCode()}.</li>
 * </ul>
 *
 * @see AbstractSearch
 * @see ArrayCfg
 * @see Board
 * @author Brandon Mejia
 * @version 2025-10-15
 */
public interface Ilayout
{
    /**
     * Generates all valid successor states (children) reachable from the current state
     * in a single step.
     *
     * @return A {@link List} of {@link Ilayout} objects, where each object is a new state
     *         reachable from the current one. If no successors are possible, an empty list
     *         should be returned.
     */
    List<Ilayout> children();

    /**
     * Checks if the current state is the goal state by comparing it to a given layout.
     *
     * @param l The goal layout to compare against.
     * @return {@code true} if the current layout is the goal, {@code false} otherwise.
     */
    boolean isGoal(Ilayout l);


    /**
     * Gets the cost of the single step (the "move") that was taken to reach this state
     * from its parent. This is the step cost `c(n, n')`, not the total path cost `g(n)`.
     * <p>
     * For the initial state of the search, this cost should be 0.
     *
     * @return The cost of the last action taken to produce this state.
     */
    double getK();

    /**
     * Calculates the heuristic estimate of the cost from this state to the goal.
     * This is the `h(n)` function in algorithms like A*.
     *
     * @param goal The goal layout.
     * @return The estimated cost to reach the goal.
     */
    double getH(Ilayout goal);

    /**
     * Provides a string representation of the layout.
     * <p>
     * This method is essential for displaying the state in the solution output and for debugging.
     * It is implicitly required by the problem specifications that rely on `System.out.println(object)`.
     *
     * @return A string representation of this layout.
     */
    @Override
    String toString();
}
