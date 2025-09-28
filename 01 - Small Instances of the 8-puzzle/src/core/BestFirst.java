package core;

import java.util.*;

/**
 * Implements the Best-First search algorithm.
 * This class is the "Context" in the Strategy design pattern.
 *
 * @author Brandon Mejia
 * @version 2025-09-07
 */
public class BestFirst
{
    protected Queue<State> abertos;
    private Map<Ilayout, State> fechados;
    private State actual;
    private Ilayout objective;


//--------------------------------------------------------------------------------------------------------------------------------
    /**
     * Represents a node in the search tree.
     */
    public static class State
    {
        private final Ilayout layout;
        private final State father;
        private final double g; // Cost of the path from the start

        public State(Ilayout l, State n) {
            layout = l;
            father = n;
            if (father != null)
                g = father.g + layout.getG();
            else g = 0.0;
        }

        public String toString() { return layout.toString(); }
        public double getG() { return g; }
        public Ilayout getLayout() { return layout; }
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
//--------------------------------------------------------------------------------------------------------------------------------
    final private List<State> sucessores(State n)
    {
        List<State> sucs = new ArrayList<>();
        List<Ilayout> children = n.layout.children();
        for (Ilayout e : children)
            sucs.add(new State(e, n));

        return sucs;
    }

    /**
     * Executes the search to find a path between an initial state and a goal.
     * @param s The initial layout.
     * @param goal The goal layout.
     * @return An iterator over the states of the solution path, or null if no solution is found.
     */
    final public Iterator<State> solve(Ilayout s, Ilayout goal)
    {
        objective = goal;
        abertos = new PriorityQueue<>(10, Comparator.comparingDouble(State::getG));
        fechados = new HashMap<>();
        abertos.add(new State(s, null));

        while (!abertos.isEmpty())
        {
            actual = abertos.poll();

            if (actual.getLayout().isGoal(objective))
            {
                // Solution found! Reconstruct the solution path.
                LinkedList<State> path = new LinkedList<>();
                State current = actual;
                while (current != null) {
                    path.addFirst(current);// funciona como stack, só que melhor O(1) em adicionar ao inicio
                    current = current.getFather();
                }
                return path.iterator();
            }

            fechados.put(actual.getLayout(), actual);
            List<State> sucs = sucessores(actual);

            for (State sucessor : sucs)
            {
                if (!fechados.containsKey(sucessor.getLayout()) && !abertos.contains(sucessor)) {
                    abertos.add(sucessor);
                }
            }
        }
        return null; // No solution found
    }
}