package core;

import java.util.*;

/**
 * Implements the Template Method design pattern for graph search algorithms.
 * This abstract class defines the skeleton of a search algorithm in the `solve` method,
 * leaving the specific strategy for ordering nodes (the "fringe") to be implemented
 * by subclasses through the `createFringe` method.
 *
 * @author Brandon Mejia
 * @preConditions
 *                 - Subclasses must implement the `createFringe()` method to define a search strategy.
 *                 - The `solve` method must be called with valid, non-null initial and goal layouts.
 * @postConditions
 *                  - The `solve` method returns an iterator for the solution path if one is found, or null otherwise.
 *                  - The state of the search (open/closed lists) is managed internally.
 *
 *
 * @see Ilayout
 * @see State
 *
 * @version 2025-09-27
 */
public abstract class AbstractSearch
{
    protected Queue<State> abertos;
    protected Map<Ilayout, State> fechados;
    protected State actual;
    protected Ilayout objective;

    /**
     * Represents a node in the search tree, wrapping a layout and maintaining search-related information.
     * Each state holds a reference to its parent, allowing for path reconstruction, and stores the
     * accumulated cost (`g`) from the initial state to this node.
     */
    public static class State
    {
        private final Ilayout layout;
        private final State father;
        private final double g; // Accumulated cost of the path from the start to this state

        /**
         * Constructs a search node.
         * The accumulated cost `g` is calculated by adding the parent's cost
         * to the cost of the step leading to the current layout.
         * @param l The layout for this state.
         * @param n The parent state in the search tree.
         */
        public State(Ilayout l, State n) {
            layout = l;
            father = n;
            if (father != null)
                g = father.g + layout.getK(); // Use getK() from Ilayout for step cost
            else g = 0.0;
        }

        @Override
        public String toString() {
            return layout.toString();
        }

        /**
         * Gets the total accumulated cost from the initial state to this state.
         * This is used by the Main class to print the final solution cost.
         * @return The total path cost (g).
         */
        public double getK() {
            return g;
        }

        /**
         * Gets the total accumulated cost from the initial state to this state.
         * This is used internally by search strategies (e.g., in a PriorityQueue) for ordering.
         * @return The total path cost (g).
         */
        public double getG() {
            return g;
        }

        /** @return The layout of this state. */
        public Ilayout getLayout() { return layout; }
        /** @return The parent state in the search path. */
        public State getFather() { return father; }

        @Override
        public int hashCode() { return layout.hashCode(); }

        @Override
        public boolean equals(Object o)
        {
            if (o == null || this.getClass() != o.getClass()) return false;
            State n = (State) o;
            return this.layout.equals(n.layout);
        }
    }

    /**
     * Generates the successor states for a given search node.
     * This method is common to most graph search algorithms.
     * @param n The parent node.
     * @return A list of `State` objects representing the children.
     */
    protected final List<State> generateSuccessors(State n)
    {
        List<State> sucs = new ArrayList<>();
        List<Ilayout> children = n.layout.children();

        for (Ilayout e : children)
            sucs.add(new State(e, n));
        return sucs;
    }

    /**
     * The "Template Method". It defines the invariant skeleton of the search algorithm.
     * It uses the `createFringe` hook method to allow subclasses to define
     * the search strategy.
     * @param s The initial layout of the problem.
     * @param goal The goal layout of the problem.
     * @return An iterator over the states of the solution path, or null if no solution is found.
     */
    public final Iterator<State> solve(Ilayout s, Ilayout goal)
    {
        objective = goal;
        abertos = createFringe(); // Hook method: subclasses define the fringe type/ordering
        fechados = new HashMap<>();
        abertos.add(new State(s, null));

        while (!abertos.isEmpty())
        {
            actual = abertos.poll();

            if (fechados.containsKey(actual.getLayout())) { // Skip if already processed via a cheaper path
                continue;
            }

            if (actual.getLayout().isGoal(objective)) {
                LinkedList<State> path = new LinkedList<>();
                State current = actual;
                while (current != null) {
                    path.addFirst(current);
                    current = current.getFather();
                }
                return path.iterator();
            }

            fechados.put(actual.getLayout(), actual);
            List<State> sucs = generateSuccessors(actual);

            for (State successor : sucs)
                if (!fechados.containsKey(successor.getLayout())) // Only add if not already in closed list
                    abertos.add(successor);

        }
        return null; // No solution found
    }

    /**
     * The "hook" method that subclasses must implement.
     * This method defines the search strategy by providing a specific type of queue (fringe).
     * @return A Queue implementation that dictates the search order.
     */
    protected abstract Queue<State> createFringe();
}