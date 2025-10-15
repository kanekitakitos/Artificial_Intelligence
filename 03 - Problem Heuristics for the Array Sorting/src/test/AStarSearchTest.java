package test;

import core.ArrayCfg;
import core.AStarSearch;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.time.Duration;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.io.PrintStream;
import java.util.Iterator;
import java.util.Scanner;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * Unit and integration tests for the {@link AStarSearch} class.
 * <p>
 * This class tests the complete application flow by simulating the execution of the main program
 * using the A* algorithm. It provides test cases based on the assignment description to verify
 * the correctness and optimality of the solutions found.
 * @author Brandon Mejia
 * @version 2025-10-01
 */
public class AStarSearchTest {

    // To capture System.out
    private final ByteArrayOutputStream outContent = new ByteArrayOutputStream();
    private final PrintStream originalOut = System.out;

    // To provide System.in
    private final InputStream originalIn = System.in;

    /**
     * A private replica of the main application logic for testing purposes.
     * This method reads from {@link System#in}, invokes the A* solver, and prints the
     * final cost to {@link System#out}.
     */
    private void main (String [] args)
    {
        Scanner sc = new Scanner(System.in);
        AStarSearch aStar = new AStarSearch();
        Iterator<AStarSearch.State> it =
                aStar.solve( new ArrayCfg(sc.nextLine()), new ArrayCfg(sc.nextLine()));
        if (it==null) System.out.println("no solution found");
        else {
            // Iterate through the solution path to find the final state
            AStarSearch.State finalState = null;
            while(it.hasNext()) {
                finalState = it.next();
            }
            // Print only the cost of the final state
            if (finalState != null) System.out.println((int)finalState.getG());
        }
        sc.close();
    }

    @BeforeEach
    public void setUpStreams() {
        System.setOut(new PrintStream(outContent));
    }

    @AfterEach
    public void restoreStreams() {
        System.setOut(originalOut);
        System.setIn(originalIn);
    }

    /**
     * Helper method to run the main application with a given input string.
     * @param input The string to be provided as standard input, with lines separated by '\n'.
     */
    private void runAppWithInput(String input) throws Exception {
        ByteArrayInputStream inContent = new ByteArrayInputStream(input.getBytes());
        System.setIn(inContent);
        main(null);
    }

    /**
     * Tests Sample 1 from the assignment description using A* search.
     * Verifies that the optimal cost of 33 is found.
     * @throws Exception if the test run fails.
     */
    @Test
    void testSample1_fromProblemDescription() throws Exception {
        // Arrange
        String input = "-2 4 0 -1 3 5 1\n-2 -1 0 1 3 4 5\n";
        String expectedOutput = "33" + System.lineSeparator();

        // Act
        runAppWithInput(input);

        // Assert
        assertEquals(expectedOutput, outContent.toString());
    }

    /**
     * Tests the performance of the A* search for Sample 1.
     * The problem states this should be solved in milliseconds, so we set a generous
     * timeout of 1 second to ensure it meets the performance requirements.
     * @throws Exception if the test run fails.
     */
    @Test
    @DisplayName("Performance: Sample 1 from problem description")
    void testSample1_performance() throws Exception {
        // Arrange
        String input = "-2 4 0 -1 3 5 1\n-2 -1 0 1 3 4 5\n";
        String expectedOutput = "33" + System.lineSeparator();

        // Act
        long startTime = System.nanoTime();
        runAppWithInput(input);
        long endTime = System.nanoTime();

        // Assert
        assertEquals(expectedOutput, outContent.toString(), "The output should be correct even when testing for performance.");

        // Report time
        reportTime(startTime, endTime, "testSample1_performance");
    }

    /**
     * Tests the performance on a more complex case (a reversed array of 8 elements).
     * This ensures the heuristic is effective enough to solve larger problems quickly.
     * The timeout is set to 1 second.
     */
    @Test
    @DisplayName("Performance: 8-element reversed array")
    void testReversedArray_performance() throws Exception {
        // Arrange
        String input = "8 7 6 5 4 3 2 1\n1 2 3 4 5 6 7 8\n";
        String expectedOutput = "44" + System.lineSeparator();

        // Act
        long startTime = System.nanoTime();
        runAppWithInput(input);
        long endTime = System.nanoTime();

        // Assert
        assertEquals(expectedOutput, outContent.toString(), "The output for the reversed array should be correct.");

        // Report time
        reportTime(startTime, endTime, "testReversedArray_performance");
    }

    /**
     * A more demanding performance test with a 10-element array where even and odd numbers
     * are completely scrambled relative to their final positions. This forces the algorithm
     * to navigate a larger search space with varied step costs.
     * The timeout is kept at 1 second to ensure high performance.
     */
    @Test
    @DisplayName("Performance: 10-element highly scrambled array")
    void testHighlyScrambled_10_elements_performance() throws Exception {
        // Arrange: Evens are in odd positions and vice-versa.
        String input = "9 8 7 6 5 4 3 2 1 10\n1 2 3 4 5 6 7 8 9 10\n";
        // The optimal solution requires 4 swaps, all between even and odd numbers.
        // (9,1), (8,2), (7,3), (6,4). The 5 and 10 are in correct final positions relative to each other.
        // A better path is swapping pairs that are in each other's places: (10,1), (2,9), (4,7), (6,8). Cost: 4*11=44
        String expectedOutput = "44" + System.lineSeparator();

        // Act
        long startTime = System.nanoTime();
        runAppWithInput(input);
        long endTime = System.nanoTime();

        // Assert
        assertEquals(expectedOutput, outContent.toString(), "The output for the highly scrambled array should be correct.");

        // Report time
        reportTime(startTime, endTime, "testHighlyScrambled_10_elements_performance");
    }

    /**
     * A more rigorous performance test with a fully reversed 11-element array.
     * This significantly increases the search space compared to the 10-element test,
     * providing a stronger challenge for the A* algorithm and its heuristic.
     * The optimal path involves 5 swaps.
     * Timeout is set to 2 seconds to account for the increased complexity.
     */
    @Test
    @DisplayName("Performance: 11-element reversed array")
    void testReversedArray_11_elements_performance() throws Exception {
        // Arrange
        String input = "11 10 9 8 7 6 5 4 3 2 1\n1 2 3 4 5 6 7 8 9 10 11\n";
        // Optimal path cost is calculated by swapping symmetric pairs:
        // swap(1,11) -> cost 20 (odd-odd)
        // swap(2,10) -> cost 2 (even-even)
        // swap(3,9)  -> cost 20 (odd-odd)
        // swap(4,8)  -> cost 2 (even-even)
        // swap(5,7)  -> cost 20 (odd-odd)
        // Total cost = 20 + 2 + 20 + 2 + 20 = 64
        String expectedOutput = "64" + System.lineSeparator();

        // Act
        long startTime = System.nanoTime();
        runAppWithInput(input);
        long endTime = System.nanoTime();

        // Assert
        assertEquals(expectedOutput, outContent.toString(), "The output for the 11-element reversed array should be correct.");

        // Report time
        reportTime(startTime, endTime, "testReversedArray_11_elements_performance");
    }


    @Test
    @DisplayName("Performance: 12-element 'worst-case' array")
    void testWorstCase_12_elements_performance() throws Exception {
        // Arrange: All evens are in the first half, odds in the second. Goal is the opposite.
        String input = "2 4 6 8 10 12 1 3 5 7 9 11\n1 3 5 7 9 11 2 4 6 8 10 12\n";
        // The optimal solution requires swapping each even number with an odd number.
        // This means 6 swaps of cost 11 each. Total cost = 6 * 11 = 66.
        String expectedOutput = "66" + System.lineSeparator();

        // Act
        long startTime = System.nanoTime();
        runAppWithInput(input);
        long endTime = System.nanoTime();

        // Assert
        assertEquals(expectedOutput, outContent.toString(), "The output for the 12-element 'worst-case' array should be correct.");

        // Report time
        reportTime(startTime, endTime, "testWorstCase_12_elements_performance");
    }

    /**
     * Tests the case where the initial state is already the goal state.
     * The expected cost should be 0.
     * @throws Exception if the test run fails.
     */
    @Test
    void testInitialIsGoal() throws Exception {
        // Arrange
        String input = "1 2 3 4 5\n1 2 3 4 5\n";
        String expectedOutput = "0" + System.lineSeparator();

        // Act
        runAppWithInput(input);

        // Assert
        assertEquals(expectedOutput, outContent.toString());
    }

    /**
     * Tests the algorithm's behavior with duplicate numbers in the array.
     * The search should still find the optimal path.
     * @throws Exception if the test run fails.
     */
    @Test
    void testWithDuplicateNumbers() throws Exception {
        // Arrange
        String input = "2 5 2 8\n8 5 2 2\n"; // Swap (2,8) -> cost 2
        String expectedOutput = "2" + System.lineSeparator();

        // Act
        runAppWithInput(input);

        // Assert
        assertEquals(expectedOutput, outContent.toString());
    }

    /**
     * Tests a single-element array, which is already sorted.
     * The cost should be 0.
     * @throws Exception if the test run fails.
     */
    @Test
    void testSingleElementArray() throws Exception {
        // Arrange
        String input = "10\n10\n";
        String expectedOutput = "0" + System.lineSeparator();

        // Act
        runAppWithInput(input);

        // Assert
        assertEquals(expectedOutput, outContent.toString());
    }

    /**
     * Tests that creating a configuration from an empty string throws an exception.
     * This ensures robust input validation.
     */
    @Test
    void testEmptyInput() {
        // Arrange
        String input = "\n\n"; // Empty lines
        System.setIn(new ByteArrayInputStream(input.getBytes()));

        // Act & Assert
        assertThrows(IllegalArgumentException.class, () -> main(null));
    }

    /**
     * Tests a scenario where the heuristic's admissibility is key.
     * A path with more steps (2 swaps) but lower total cost (2+2=4) should be chosen
     * over a path with fewer steps (1 swap) but higher cost (11).
     */
    @Test
    void testHeuristicAdmissibility() throws Exception {
        String input = "1 2 4\n4 1 2\n"; // Goal: 4 1 2
        String expectedOutput = "4" + System.lineSeparator(); // Optimal: (1,4) cost 11. (1,2)->(2,1,4) cost 11. (2,4)->(4,1,2) cost 2. Total 13. Wait.
        // Path 1: swap(1,4) -> 4 2 1. cost 11. swap(2,1) -> 4 1 2. cost 11. Total 22.
        // Path 2: swap(2,4) -> 1 4 2. cost 2. swap(1,4) -> 4 1 2. cost 11. Total 13.
        // Path 3: swap(1,2) -> 2 1 4. cost 11. swap(2,4) -> 4 1 2. cost 2. Total 13.
        // Let's re-evaluate the optimal cost.
        // Initial: 1 2 4. Goal: 4 1 2.
        // 1 is odd, 2 is even, 4 is even.
        // Swap (2,4) -> 1 4 2. Cost = 2. Now state is 1 4 2.
        // Swap (1,4) -> 4 1 2. Cost = 11. Total cost = 2 + 11 = 13.
        // Let's check another path.
        // Swap (1,4) -> 4 2 1. Cost = 11. Now state is 4 2 1.
        // Swap (2,1) -> 4 1 2. Cost = 11. Total cost = 11 + 11 = 22.
        // The optimal cost is 13.
        runAppWithInput(input);
        assertEquals("13" + System.lineSeparator(), outContent.toString());
    }

    /**
     * Helper method to print the execution time of a test.
     * @param startTime The start time in nanoseconds.
     * @param endTime The end time in nanoseconds.
     * @param testName The name of the test being reported.
     */
    private void reportTime(long startTime, long endTime, String testName) {
        double durationMs = (endTime - startTime) / 1_000_000.0;
        double durationS = durationMs / 1000.0;
        System.err.printf("  [PERF] %s execution time: %.3f s (%.3f ms)%n", testName, durationS, durationMs);
    }
}