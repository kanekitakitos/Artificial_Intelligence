package core;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;


/**
 * Represents a state (configuration) in the array sorting problem.
 * This class implements the {@link Ilayout} interface, providing the specific logic
 * for generating children (by swapping elements), calculating costs, and checking for the goal state.
 *
 * @preConditions
 *                 - The constructor `ArrayCfg(String)` must be called with a string of space-separated integers.
 * @postConditions
 *                  - An immutable `ArrayCfg` object is created.
 *                  - The object can generate its children, be compared to a goal, and provide its step cost.
 *
 * @see Ilayout
 *
 * @author Brandon Mejia
 * @version 2025-09-27
 */
public class ArrayCfg implements Ilayout
{
    private final int[] data;
    private final double cost; // Cost to generate this state from its parent
    /**
     * Constructs an array configuration from a string of space-separated integers.
     * This constructor is used for the initial and goal states.
     *
     * @preConditions
     *                 - The input string `s` must contain valid integers separated by spaces.
     *                 - Non-integer values will cause a `NumberFormatException`.
     * @postConditions
     *                  - A new `ArrayCfg` object is created with the specified integer configuration.
     *                  - The step cost (`cost`) is initialized to 0.
     *
     * @param s The string representing the array configuration.
     */
    public ArrayCfg(String s)
    {
        this.data = Arrays.stream(s.split(" ")).mapToInt(Integer::parseInt).toArray();
        this.cost = 0; // Initial state has no parent, cost is 0
    }

    /**
     * Private constructor for creating successor states (children).
     * @param data The new integer array for the child state.
     * @param cost The cost of the swap that generated this child state.
     */
    private ArrayCfg(int[] data, double cost) {
        this.data = data;
        this.cost = cost;
    }

    /**
     * Generates all valid successor states by swapping pairs of elements.
     * The swapping order is from left to right, as specified in the problem statement.
     * @return A list of new `ArrayCfg` objects representing all possible next states.
     */
    @Override
    public List<Ilayout> children()
    {
        List<Ilayout> children = new ArrayList<>();
        for (int i = 0; i < data.length - 1; i++) {
            for (int j = i + 1; j < data.length; j++) {
                int[] childData = Arrays.copyOf(data, data.length);

                // Swap elements
                int temp = childData[i];
                childData[i] = childData[j];
                childData[j] = temp;

                // Calculate cost
                double swapCost = calculateCost(data[i], data[j]);

                children.add(new ArrayCfg(childData, swapCost));
            }
        }
        return children;
    }

    /**
     * Calculates the cost of swapping two integers based on their parity.
     * @param a The first integer.
     * @param b The second integer.
     * @return The cost of the swap (2, 11, or 20).
     */
    private double calculateCost(int a, int b)
    {
        boolean isAEven = a % 2 == 0;
        boolean isBEven = b % 2 == 0;

        if (isAEven && isBEven)
            return 2.0;

         else if (!isAEven && !isBEven)
            return 20;

         else
            return 11;

    }

    /**
     * Checks if this configuration is the goal state by comparing it to another layout.
     * @param l The goal layout to compare against.
     * @return `true` if this configuration is identical to the goal layout, `false` otherwise.
     */
    @Override
    public boolean isGoal(Ilayout l)
    {
        return this.equals(l);
    }

    /**
     * Gets the cost of the single swap that led to this configuration.
     * @return The cost of the move from the parent state.
     */
    @Override
    public double getK()
    {
        return this.cost;
    }

    /**
     * Returns a string representation of the array for display.
     * Integers are separated by spaces.
     * @return A formatted string of the array configuration.
     */
    @Override
    public String toString()
    {
        return Arrays.stream(data)
                .mapToObj(String::valueOf)
                .collect(Collectors.joining(" "));
    }
    /**
     * Compares this configuration with another object for equality.
     * @param o The object to compare with.
     * @return `true` if the other object is an `ArrayCfg` with the exact same integer array.
     */
    @Override
    public boolean equals(Object o)
    {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        ArrayCfg arrayCfg = (ArrayCfg) o;
        return Arrays.equals(data, arrayCfg.data);
    }

    /**
     * Computes the hash code for this configuration.
     * The hash code is based on the contents of the integer array.
     * @return The hash code for this configuration.
     */
    @Override
    public int hashCode() {
        return Arrays.hashCode(data);
    }
}
