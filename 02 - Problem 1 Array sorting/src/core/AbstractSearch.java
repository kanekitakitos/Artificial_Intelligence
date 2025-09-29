package core;

import java.util.*;

/**
 * Deterministic uniform-cost search base class.
 *
 * Key change: replace a plain PriorityQueue with a TreeMap-based bucketed queue:
 *   TreeMap<Double, ArrayDeque<State>>
 *
 * This enforces:
 *  - processing by increasing g (cumulative cost),
 *  - FIFO within equal-g groups (insert at tail),
 *  - lazy removal of obsolete states (skip when polled if not the current best in abertosMap).
 *
 * This yields deterministic behavior required by the sample outputs.
 */
public abstract class AbstractSearch {
    // kept for compatibility; we use our own fringe internally in solve()
    protected Queue<State> abertos;
    protected Map<Ilayout, State> fechados; // expanded states
    protected State actual;
    protected Ilayout objective;
    private static long sequenceCounter = 0;

    public static class State {
        private final Ilayout layout;
        private final State father;
        private final double g; // cost from start
        private final long sequenceNumber;

        public State(Ilayout l, State n) {
            layout = l;
            father = n;
            sequenceNumber = AbstractSearch.sequenceCounter++;
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
        public long getSequenceNumber() { return sequenceNumber; }
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

    protected final List<State> generateSuccessors(State n) {
        List<State> sucs = new ArrayList<>();
        List<Ilayout> children = n.layout.children();
        for (Ilayout e : children) {
            sucs.add(new State(e, n));
        }
        return sucs;
    }

    /**
     * Note: createFringe is left abstract for compatibility, but the solve() method
     * below uses an internal bucketed structure for determinism.
     */
    protected abstract Queue<State> createFringe();

    // dentro do teu AbstractSearch (substitui o solve atual)
    public final Iterator<State> solve(Ilayout s, Ilayout goal) {
        objective = goal;
        fechados = new HashMap<>();
        Map<Ilayout, State> abertosMap = new HashMap<>();

        // Bucketed fringe: key = cumulative g (long), value = FIFO deque of States
        TreeMap<Long, ArrayDeque<State>> buckets = new TreeMap<>();

        sequenceCounter = 0L;
        // initial
        State initial = new State(s, null); // adapt State.g to be long internally if you change it
        // ensure initial.g == 0
        long g0 = Math.round(initial.getG());
        buckets.computeIfAbsent(g0, k -> new ArrayDeque<>()).addLast(initial);
        abertosMap.put(initial.getLayout(), initial);

        while (!buckets.isEmpty()) {
            Map.Entry<Long, ArrayDeque<State>> entry = buckets.firstEntry();
            ArrayDeque<State> dq = entry.getValue();

            State polled = null;
            // pop from head until we find a non-obsolete state
            while (!dq.isEmpty()) {
                State candidate = dq.peekFirst();
                State currentBest = abertosMap.get(candidate.getLayout());
                if (currentBest == candidate) {
                    polled = dq.removeFirst();
                    abertosMap.remove(polled.getLayout()); // will be expanded now
                    break;
                } else {
                    // obsolete state, drop it
                    dq.removeFirst();
                }
            }
            // if this bucket is empty now, remove it
            if (dq.isEmpty()) buckets.remove(entry.getKey());

            if (polled == null) continue; // all entries were obsolete, move to next bucket

            actual = polled;

            // goal test
            if (actual.getLayout().isGoal(objective)) {
                LinkedList<State> path = new LinkedList<>();
                State cur = actual;
                while (cur != null) { path.addFirst(cur); cur = cur.getFather(); }
                return path.iterator();
            }

            // mark closed
            fechados.put(actual.getLayout(), actual);

            // expand successors
            List<State> sucs = generateSuccessors(actual); // IMPORTANT: ensure this uses j descending
            for (State succ : sucs) {
                if (fechados.containsKey(succ.getLayout())) continue; // already expanded

                long succG = Math.round(succ.getG()); // ensure succ.g computed as integer-like

                State existing = abertosMap.get(succ.getLayout());
                if (existing == null || succG < Math.round(existing.getG())) {
                    // insert at tail of bucket succG
                    buckets.computeIfAbsent(succG, k -> new ArrayDeque<>()).addLast(succ);
                    abertosMap.put(succ.getLayout(), succ);
                }
            }
        }

        return null; // no solution
    }

}
