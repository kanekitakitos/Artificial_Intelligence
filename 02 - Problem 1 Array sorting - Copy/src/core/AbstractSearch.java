
package core;

import java.util.*;

/**
 * An abstract base class for implementing state-space search algorithms using the Template Method pattern.
 * <p>
 * This class provides the core search loop, management of open (fringe) and closed lists,
 * and path reconstruction. The `fechados` (closed) list is implemented as a map that not only
 * tracks visited states but also stores the best path found so far to each state, enabling
 * efficient pruning and path optimization.
 *
 * <h3>Implementing a Search Strategy</h3>
 * To implement a specific search algorithm, extend this class and provide an implementation for the
 * {@link #createFringe()} method. The type of {@link Queue} returned determines the search strategy.
 *
 * <h4>Uniform-Cost Search (UCS)</h4>
 * <p>
 * Expands the node with the lowest path cost (g). Guarantees the least-cost path.
 * </p>
 * <ul>
 *   <li><b>Data Structure:</b> {@link PriorityQueue}</li>
 *   <li><b>Comparator:</b> Order by `State::getG`. For stable sorting (FIFO for same-cost nodes), use a
 *       secondary comparison on the state's sequence ID.
 *       <pre>{@code
 *       Comparator<State> comparator = Comparator.comparingDouble(State::getG)
 *                                                .thenComparingLong(State::getSequenceId);
 *       return new PriorityQueue<>(comparator);
 *       }</pre>
 *   </li>
 * </ul>
 *
 * <h4>Breadth-First Search (BFS)</h4>
 * <p>
 * Expands nodes level by level. Guarantees the shortest path in terms of number of steps,
 * assuming all step costs are equal.
 * </p>
 * <ul>
 *   <li><b>Data Structure:</b> A simple FIFO queue, like {@link LinkedList}.</li>
 *   <li><b>Implementation:</b>
 *       <pre>{@code return new LinkedList<>();}</pre>
 *   </li>
 * </ul>
 *
 * <h4>Depth-First Search (DFS)</h4>
 * <p>
 * Expands the deepest node first. It's not optimal and may not find a solution if the
 * search space has infinite branches.
 * </p>
 * <ul>
 *   <li><b>Data Structure:</b> A LIFO stack, which can be implemented with {@link ArrayDeque}.</li>
 *   <li><b>Implementation:</b> Since the `solve` method uses `queue.add(e)`, which adds to the end,
 *       we must override it to add to the front to simulate a stack's `push` operation.
 *       <pre>{@code
 *       return new ArrayDeque<>() {
 *           @Override
 *           public boolean add(State state) {
 *               this.addFirst(state);
 *               return true;
 *           }
 *       };
 *       }</pre>
 *   </li>
 * </ul>
 *
 * @preConditions Subclasses must implement {@link #createFringe()}.
 * @postConditions The {@link #solve} method returns an iterator for the solution path or null if no solution is found.
 *
 * @see Ilayout
 * @see State
 *
 * @author Brandon Mejia
 * @version 2025-09-30
 */
public abstract class AbstractSearch
{
    protected Queue<State> abertos;
    protected Map<Ilayout, State> fechados; // Using this map to track the best state found for a layout
    protected State actual;
    protected Ilayout objective;
    private static long sequenceCounter = 0;

    /**
     * Represents a node in the search space. It contains the layout (the actual state),
     * a reference to its parent node (for path reconstruction), the accumulated cost (`g`)
     * from the start node, and a unique sequence ID for tie-breaking.
     */
    public static class State {
        private final Ilayout layout;
        private final State father;
        private final double g; // Cost from start to current state
        private final long sequenceId;

        /**
         * Constructs a new State.
         * <p>
         * The accumulated cost `g` is calculated based on the parent's cost plus the
         * cost of the step leading to this state. The initial state has a cost of 0.
         * @param l The layout of this state.
         * @param n The parent state (null for the initial state).
         */
        public State(Ilayout l, State n) {
            layout = l;
            father = n;
            sequenceId = AbstractSearch.sequenceCounter++;

            if (father != null) {
                g = father.g + layout.getK();
            } else {
                g = 0.0;
            }
        }

        @Override
        public String toString() {
            return layout.toString();
        }

        public double getK() { return g; }
        public double getG() { return g; }
        public long getSequenceId() { return sequenceId; }
        public Ilayout getLayout() { return layout; }
        public State getFather() { return father; }

        @Override
        public int hashCode() { return layout.hashCode(); }

        @Override
        public boolean equals(Object o) {
            if (o == null || getClass() != o.getClass()) return false;
            State n = (State) o;
            return this.layout.equals(n.layout);
        }
    }

    /**
     * Generates all successor states for a given state.
     *
     * @param n The state to expand.
     * @return A list of successor states.
     */
    protected final List<State> generateSuccessors(State n)
    {
        List<State> sucs = new ArrayList<>();
        List<Ilayout> children = n.layout.children();
        for (Ilayout e : children) {
            sucs.add(new State(e, n));
        }
        return sucs;
    }

    /**
     * Executes the search algorithm to find a path from an initial state to a goal state.
     *
     * @param s    The initial layout.
     * @param goal The goal layout.
     * @return An {@link Iterator} for the solution path if found; otherwise, {@code null}.
     */
    public final Iterator<State> solve(Ilayout s, Ilayout goal)
    {
        objective = goal;
        abertos = createFringe();
        fechados = new HashMap<>();
        sequenceCounter = 0;

        State initialState = new State(s, null);
        abertos.add(initialState);
        fechados.put(initialState.getLayout(), initialState);

        while (!abertos.isEmpty())
        {
            actual = abertos.poll();


            State currentBest = fechados.get(actual.getLayout());
            if (actual.getG() > currentBest.getG())
            {
              //  System.err.println("SKIPPING stale node=" + actual.getLayout() + " seq=" + actual.getSequenceId() + " totalG=" + actual.getG() + " (better path exists with G=" + currentBest.getG() + ")");
                continue;
            }

            //System.err.println("--------------------------------------------------------------------------------------------------------------------");
            //System.err.println("EXPANDING node=" + actual.getLayout() + " totalG=" + actual.getG() + " seq=" + actual.getSequenceId());

            if (actual.getLayout().isGoal(objective)) {
                //System.err.println("GOAL FOUND!");
                LinkedList<State> path = new LinkedList<>();
                State current = actual;
                while (current != null) {
                    path.addFirst(current);
                    current = current.getFather();
                }
                return path.iterator();
            }

            List<State> sucs = generateSuccessors(actual);
            int childNum = 0;
            for (State successor : sucs) {
                State existing = fechados.get(successor.getLayout());


                if (existing == null || successor.getG() < existing.getG())
                {
                    fechados.put(successor.getLayout(), successor);
                    abertos.add(successor);
                    //System.err.println("  CHILD#" + childNum + " ADD child= " + successor.getLayout() + " step= " + successor.getLayout().getK() + " totalG= " + successor.getG() + " seq=" + successor.getSequenceId());
                } //else {
                  //  System.err.println("  CHILD#" + childNum + " IGNORE = " + successor.getLayout() + " step= " + successor.getLayout().getK() + " totalG= " + successor.getG() + " seq=" + successor.getSequenceId() + " (worse than existing seq=" + existing.getSequenceId() + " existingG=" + existing.getG() + ")");
                //}
                childNum++;
            }
        }
        return null;
    }

    /**
     * Factory method for creating the fringe (the "abertos" list).
     * The data structure returned by this method defines the search strategy.
     * @return A {@link Queue} instance to be used as the fringe.
     */
    protected abstract Queue<State> createFringe();
}