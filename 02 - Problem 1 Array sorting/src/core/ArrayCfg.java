package core;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Immutable array configuration implementing Ilayout.
 */
public final class ArrayCfg implements Ilayout {

    private final int[] data;
    private final int cost; // cost of the swap that produced this state (0 for initial)

    // Constructor used for initial/goal states (from a string).
    public ArrayCfg(String s) {
        if (s == null) throw new IllegalArgumentException("Input string cannot be null");
        String trimmed = s.trim();
        if (trimmed.isEmpty()) {
            this.data = new int[0];
        } else {
            // split on any whitespace sequence (handles multiple spaces, tabs, etc.)
            String[] parts = trimmed.split("\\s+");
            int[] parsed = Arrays.stream(parts).mapToInt(Integer::parseInt).toArray();
            this.data = Arrays.copyOf(parsed, parsed.length); // defensive copy
        }
        this.cost = 0;
    }

    // Private constructor for children
    private ArrayCfg(int[] data, int cost) {
        this.data = data; // already a copy by caller
        this.cost = cost;
    }

    @Override
    public List<Ilayout> children() {
        int n = data.length;
        if (n < 2) return Collections.emptyList();

        List<Ilayout> children = new ArrayList<>(n * (n - 1) / 2);
        for (int i = 0; i < n - 1; i++) {
            for (int j = n-1; j > i ; j--) {
                int[] childData = Arrays.copyOf(data, n);

                // swap
                int tmp = childData[i];
                childData[i] = childData[j];
                childData[j] = tmp;

                // compute cost using original values from parent
                int swapCost = calculateCost(data[i], data[j]);

                children.add(new ArrayCfg(childData, swapCost));
            }
        }
        return Collections.unmodifiableList(children);
    }

    /**
     * Compute swap cost:
     * - even & even -> 2
     * - odd & odd -> 20
     * - mixed -> 11
     */
    private static int calculateCost(int a, int b) {
        // Using bitwise to check parity; works for negative numbers too.
        boolean aEven = (a & 1) == 0;
        boolean bEven = (b & 1) == 0;

        if (aEven && bEven) return 2;
        if (!aEven && !bEven) return 20;
        return 11;
    }

    @Override
    public boolean isGoal(Ilayout l) {
        if (!(l instanceof ArrayCfg)) return false;
        return Arrays.equals(this.data, ((ArrayCfg) l).data);
    }

    @Override
    public double getK() {
        return (double) this.cost;
    }

    @Override
    public String toString() {
        return Arrays.stream(data)
                .mapToObj(String::valueOf)
                .collect(Collectors.joining(" "));
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof ArrayCfg)) return false;
        ArrayCfg other = (ArrayCfg) o;
        return Arrays.equals(this.data, other.data);
    }

    @Override
    public int hashCode() {
        return Arrays.hashCode(data);
    }

    // Optional: expose a defensive copy if needed elsewhere
    public int[] asArrayCopy() {
        return Arrays.copyOf(data, data.length);
    }
}
