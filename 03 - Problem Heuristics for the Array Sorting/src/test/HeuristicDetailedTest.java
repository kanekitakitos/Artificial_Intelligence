package test;

import core.ArrayCfg;
import org.junit.jupiter.api.Test;

import java.util.*;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Detailed tests for ArrayCfg.getH with explanatory output.
 * These tests:
 *  - reconstruct the permutation -> cycles
 *  - for cycles of size k<=5 compute the exact minimum cost via brute-force enumerationg
 *  - compare the brute-force minimum with ArrayCfg.getH
 *
 * The tests print step-by-step explanations to System.out so you can follow the heuristic.
 *
 * Note: these tests duplicate the parity-cost logic (E-E=2, E-O=11, O-O=20)
 * so they are independent of ArrayCfg.calculateCost (which is private).
 */
public class HeuristicDetailedTest {

    // parity cost helper (duplicate of ArrayCfg.calculateCost)
    private static int swapCost(int a, int b) {
        boolean aEven = (a == 0) || ((a & 1) == 0);
        boolean bEven = (b == 0) || ((b & 1) == 0);
        if (aEven == bEven) return aEven ? 2 : 20;
        return 11;
    }

    // parse space separated ints
    private static int[] parse(String s) {
        return Arrays.stream(s.trim().split("\\s+")).mapToInt(Integer::parseInt).toArray();
    }

    // build targetIndex like in ArrayCfg.getH (handles duplicates using queues)
    private static int[] buildTargetIndex(int[] data, int[] goal) {
        int n = data.length;
        Map<Integer, ArrayDeque<Integer>> posMap = new HashMap<>();
        for (int j = 0; j < n; j++) posMap.computeIfAbsent(goal[j], k -> new ArrayDeque<>()).addLast(j);

        int[] targetIndex = new int[n];
        for (int i = 0; i < n; i++) {
            ArrayDeque<Integer> q = posMap.get(data[i]);
            if (q == null || q.isEmpty()) {
                throw new IllegalArgumentException("Multisets differ between data and goal");
            }
            targetIndex[i] = q.removeFirst();
        }
        return targetIndex;
    }

    // decompose into cycles (list of cycles, each cycle is list of indices)
    private static List<List<Integer>> cyclesFromTargetIndex(int[] targetIndex) {
        int n = targetIndex.length;
        boolean[] visited = new boolean[n];
        List<List<Integer>> cycles = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            if (visited[i] || targetIndex[i] == i) {
                visited[i] = true;
                continue;
            }
            List<Integer> cyc = new ArrayList<>();
            int cur = i;
            while (!visited[cur]) {
                visited[cur] = true;
                cyc.add(cur);
                cur = targetIndex[cur];
            }
            if (cyc.size() > 0) cycles.add(cyc);
        }
        return cycles;
    }

    // brute-force minimal cost to transform cycleVals -> cycleGoalVals using exactly (k-1) swaps among cycle positions
    private static double bruteForceCycleMinCost(List<Integer> cycleVals, List<Integer> cycleGoalVals) {
        int k = cycleVals.size();
        if (k == 1) return 0.0;
        if (k == 2) {
            // only one swap needed
            return swapCost(cycleVals.get(0), cycleVals.get(1));
        }

        // build all unordered pairs among indices 0..k-1
        List<int[]> pairList = new ArrayList<>();
        for (int a = 0; a < k - 1; a++) for (int b = a + 1; b < k; b++) pairList.add(new int[]{a, b});
        int base = pairList.size();
        int steps = k - 1;
        long sequences = 1;
        for (int i = 0; i < steps; i++) sequences *= base; // upper bound enumerations

        double best = Double.POSITIVE_INFINITY;
        int[] idx = new int[steps];
        for (long seq = 0; seq < sequences; seq++) {
            int[] curVals = new int[k];
            for (int t = 0; t < k; t++) curVals[t] = cycleVals.get(t);

            double costSum = 0;
            boolean pruned = false;
            for (int step = 0; step < steps; step++) {
                int pIdx = pairList.get(idx[step])[0];
                int qIdx = pairList.get(idx[step])[1];
                int va = curVals[pIdx];
                int vb = curVals[qIdx];
                costSum += swapCost(va, vb);
                // perform swap
                int tmp = curVals[pIdx]; curVals[pIdx] = curVals[qIdx]; curVals[qIdx] = tmp;
                if (costSum >= best) { pruned = true; break; }
            }
            if (!pruned) {
                boolean matches = true;
                for (int t = 0; t < k; t++) {
                    if (curVals[t] != cycleGoalVals.get(t)) { matches = false; break; }
                }
                if (matches && costSum < best) best = costSum;
            }
            // increment mixed-radix index
            for (int p = steps - 1; p >= 0; p--) {
                idx[p]++;
                if (idx[p] < base) break;
                idx[p] = 0;
            }
        }
        if (best == Double.POSITIVE_INFINITY) {
            // should not happen; fallback: return conservative lower bound (use greedy pairing)
            int even = 0, odd = 0;
            for (int v : cycleVals) { if ((v & 1) == 0) even++; else odd++; }
            int swaps = k - 1;
            double sum = 0;
            for (int s = 0; s < swaps; s++) {
                if (even >= 2) { sum += 2; even -= 2; }
                else if (even >= 1 && odd >= 1) { sum += 11; even -= 1; odd -= 1; }
                else if (odd >= 2) { sum += 20; odd -= 2; }
                else break;
            }
            return sum;
        }
        return best;
    }

    // compute full heuristic via cycle decomposition + brute-force for small cycles and greedy for large ones
    private static double computeHeuristicIndependent(int[] data, int[] goal) {
        int n = data.length;
        int[] targetIndex = buildTargetIndex(data, goal);
        List<List<Integer>> cycles = cyclesFromTargetIndex(targetIndex);
        double total = 0.0;

        // aggregated fallback for cycles >5
        int swapsFallback = 0;
        int evenFallback = 0;
        int oddFallback = 0;

        for (List<Integer> cyc : cycles)
        {
            int k = cyc.size();
            // collect cycle values in the *cycle order used by ArrayCfg.getH* (value currently at the cycle position)
            List<Integer> vals = new ArrayList<>();
            List<Integer> goalVals = new ArrayList<>();
            for (int pos : cyc) {
                vals.add(data[pos]);
                goalVals.add(goal[pos]);
            }
            if (k == 2) {
                total += swapCost(vals.get(0), vals.get(1));
                System.out.println("2-cycle positions " + cyc + " values " + vals + " cost exact " + swapCost(vals.get(0), vals.get(1)));
            } else if (k <= 5) {
                double best = bruteForceCycleMinCost(vals, goalVals);
                total += best;
                System.out.println(k + "-cycle positions " + cyc + " values " + vals + " goalVals " + goalVals + " bestExactCost " + best);
            } else {
                // fallback
                swapsFallback += (k - 1);
                for (int v : vals) { if ((v & 1) == 0) evenFallback++; else oddFallback++; }
                System.out.println("Large-cycle (fallback) positions " + cyc + " values " + vals + " swapsNeeded " + (k - 1));
            }
        }

        // apply greedy on fallback
        for (int s = 0; s < swapsFallback; s++) {
            if (evenFallback >= 2) { total += 2; evenFallback -= 2; }
            else if (evenFallback >= 1 && oddFallback >= 1) { total += 11; evenFallback -= 1; oddFallback -= 1; }
            else if (oddFallback >= 2) { total += 20; oddFallback -= 2; }
            else break;
        }
        if (swapsFallback > 0) System.out.println("Fallback aggregated swaps " + swapsFallback + " => added greedy cost contribution; final subtotal " + total);
        return total;
    }

    // Helper to pretty-print arrays
    private static String arr(int[] a) { return Arrays.stream(a).mapToObj(String::valueOf).collect(Collectors.joining(" ")); }

    // === Tests ===

    @Test
    public void test_getH_length3_twoCycle_exact() {
        int[] init = parse("2 1 3");
        int[] goal = parse("1 2 3");
        ArrayCfg cfg = new ArrayCfg("2 1 3");

        double hIndependent = computeHeuristicIndependent(init, goal);
        double hFromClass = cfg.getH(new ArrayCfg("1 2 3"));

        System.out.println("Test length 3 (2-cycle): init=" + arr(init) + " goal=" + arr(goal));
        System.out.println("Independent h = " + hIndependent + "   class h = " + hFromClass);
        assertEquals(hIndependent, hFromClass, 1e-9);
        assertEquals(11.0, hFromClass, 1e-9); // known exact
    }

    @Test
    public void test_getH_length4_cycle_k3_example() {
        int[] init = parse("4 1 3 2");
        int[] goal = parse("1 2 3 4");
        ArrayCfg cfg = new ArrayCfg("4 1 3 2");

        double hIndependent = computeHeuristicIndependent(init, goal);
        double hFromClass = cfg.getH(new ArrayCfg("1 2 3 4"));

        System.out.println("Test length 4 (contains k=3 cycle): init=" + arr(init) + " goal=" + arr(goal));
        System.out.println("Independent h = " + hIndependent + "   class h = " + hFromClass);
        assertEquals(hIndependent, hFromClass, 1e-9);
        // we expect h to be 13 for this particular case (one k=3 cycle resolved exactly)
        assertEquals(13.0, hFromClass, 1e-9);
    }

    @Test
    public void test_getH_length5_cycle_k4_example() {
        int[] init = parse("5 2 4 1 3");
        int[] goal = parse("1 2 3 4 5");
        ArrayCfg cfg = new ArrayCfg("5 2 4 1 3");

        double hIndependent = computeHeuristicIndependent(init, goal);
        double hFromClass = cfg.getH(new ArrayCfg("1 2 3 4 5"));

        System.out.println("Test length 5 (k=4 cycle): init=" + arr(init) + " goal=" + arr(goal));
        System.out.println("Independent h = " + hIndependent + "   class h = " + hFromClass);
        assertEquals(hIndependent, hFromClass, 1e-9);
        // we don't hardcode a numeric expected here because brute-force computes it;
        // we assert equality between the independent brute-force and the class implementation.
    }

    @Test
    public void test_getH_length5_mixed_example_k4_plus_fixed() {
        int[] init = parse("5 2 3 4 1");
        int[] goal = parse("1 2 3 4 5");
        ArrayCfg cfg = new ArrayCfg("5 2 3 4 1");

        double hIndependent = computeHeuristicIndependent(init, goal);
        double hFromClass = cfg.getH(new ArrayCfg("1 2 3 4 5"));

        System.out.println("Test length 5 (another k=4 example): init=" + arr(init) + " goal=" + arr(goal));
        System.out.println("Independent h = " + hIndependent + "   class h = " + hFromClass);
        assertEquals(hIndependent, hFromClass, 1e-9);
    }

    @Test
    public void test_getH_small_random_samples() {
        // random small tests to increase confidence (k up to 5 cycles inside small arrays)
        Random rnd = new Random(12345);
        for (int iter = 0; iter < 12; iter++) {
            int n = 3 + rnd.nextInt(3); // n in [3..5]
            List<Integer> vals = new ArrayList<>();
            for (int i = 1; i <= n; i++) vals.add(i); // use 1..n distinct values
            Collections.shuffle(vals, rnd);
            int[] init = vals.stream().mapToInt(Integer::intValue).toArray();

            List<Integer> goalList = new ArrayList<>();
            for (int i = 1; i <= n; i++) goalList.add(i);
            int[] goal = goalList.stream().mapToInt(Integer::intValue).toArray();

            ArrayCfg cfg = new ArrayCfg(Arrays.stream(init).mapToObj(String::valueOf).collect(Collectors.joining(" ")));

            double hIndependent = computeHeuristicIndependent(init, goal);
            double hFromClass = cfg.getH(new ArrayCfg(Arrays.stream(goal).mapToObj(String::valueOf).collect(Collectors.joining(" "))));

            System.out.println("Random test init=" + arr(init) + " goal=" + arr(goal) + " => h_ind=" + hIndependent + " h_class=" + hFromClass);
            assertEquals(hIndependent, hFromClass, 1e-9);
        }
    }
}
