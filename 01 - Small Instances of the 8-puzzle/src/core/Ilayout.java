package core;

import java.util.List;

/**
 * Defines the contract for a state (or "layout") within a state-space search problem.
 *
 * @preConditions
 *                 - Any class implementing this interface must provide concrete implementations
 *                   for all its methods: children(), isGoal(), and getG().
 *
 * @postConditions
 *                  - An implementing class will represent a specific state of a problem.
 *                  - It will be able to generate its valid successor states (children).
 *                  - It will be able to determine if it matches a given goal state.
 *                  - It will provide the cost associated with the move that led to this state.
 *
 *                  This interface is a core component of the search framework, abstracting the
 *                  problem-specific details from the search algorithm itself. It acts as the
 *                  "Strategy" in a Strategy design pattern, where different implementations
 *                  (e.g., `Board` for the 8-puzzle) can be used by the "Context" (`BestFirst` search).
 *
 * @see BestFirst
 * @see Board
 *
 * @author Brandon Mejia
 * @version 2025-09-07
 */
public interface Ilayout
{
    /**
     * Generates all valid successor states (children) from the current layout.
     *
     * @preConditions
     *                 - The current layout must be in a valid state.
     * @postConditions
     *                  - Returns a `List` of `Ilayout` objects, where each object is a valid
     *                    next state reachable from the current state in one step.
     *                  - If no successors can be generated, an empty list is returned.
     *
     * @return The list of child layouts.
     */
    List<Ilayout> children();

    /**
     * Checks if the current layout is the goal state.
     *
     * @preConditions
     *                 - The goal layout `l` must not be null.
     * @postConditions
     *                  - Returns `true` if the properties of the current layout are identical
     *                    to the properties of the goal layout `l`.
     *                  - Returns `false` otherwise.
     *
     * @param l The goal layout to compare against.
     * @return `true` if the current layout is the goal, `false` otherwise.
     */
    boolean isGoal(Ilayout l);

    /**
     * Gets the cost of the move to reach this layout from its immediate parent.
     * This is NOT the total cost from the initial state (which is `g` in the `State` class).
     *
     * @preConditions
     *                 - None.
     * @postConditions
     *                  - Returns a non-negative `double` value representing the cost of a single step.
     *                  - For unweighted problems, this value is typically 1.0.
     *
     * @return The cost to move from the parent state to this state.
     */
    double getG();
}
