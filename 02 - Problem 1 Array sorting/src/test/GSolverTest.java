package test;

import core.ArrayCfg;
import core.GSolver;
import core.Ilayout;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;



import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.io.PrintStream;
import java.util.Iterator;
import java.util.Scanner;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Unit and integration tests for the array sorting problem solver.
 * This class tests the complete application flow by simulating the `Main` class execution.
 * It provides various test cases, including edge cases, performance, and memory usage checks.
 * @author Brandon Mejia
 * @version 2025-09-27
 */
public class GSolverTest {

    // To capture System.out
    private final ByteArrayOutputStream outContent = new ByteArrayOutputStream();
    private final PrintStream originalOut = System.out;

    // To provide System.in
    private final InputStream originalIn = System.in;

    /**
     * Before each test, redirect System.out to our stream so we can capture the output.
     */
    @BeforeEach
    public void setUpStreams() {
        System.setOut(new PrintStream(outContent));
    }

    /**
     * Simulates the main application entry point for testing purposes.
     * Reads from the redirected System.in and writes to the redirected System.out.
     * @param args Command-line arguments (not used).
     * @throws Exception if the solver throws an exception.
     */
    public void main (String [] args) throws Exception
    {
        Scanner sc = new Scanner(System.in);
        GSolver gs = new GSolver(); // GSolver now extends AbstractSearch
        Iterator<GSolver.State> it =
                gs.solve( new ArrayCfg(sc.nextLine()), new ArrayCfg(sc.nextLine()));

        if (it==null) System.out.println("no solution found");

        else
        {
            while(it.hasNext())
            {
                GSolver.State i = it.next();
                System.out.println(i);
                if (!it.hasNext()) System.out.println((int)i.getK());
            }
        }
        sc.close();
    }

    /**
     * After each test, restore the original System.out and System.in streams.
     */
    @AfterEach
    public void restoreStreams() {
        System.setOut(originalOut);
        System.setIn(originalIn);
    }

    /**
     * Helper method to run the main application with a given input string.
     * @param input The string to be provided as standard input, with lines separated by '\n'.
     * @throws Exception if the main method throws an exception.
     */
    private void runAppWithInput(String input) throws Exception {
        ByteArrayInputStream inContent = new ByteArrayInputStream(input.getBytes());
        System.setIn(inContent);
        // We need to use the Main class from the default package
        main(null);
    }

    /**
     * Tests the case where the initial state is already the goal state.
     * The cost should be 0.
     * @throws Exception if the test run fails.
     */
    @Test
    void testAlreadySorted() throws Exception {
        // Arrange
        String input = "1 2 3\n1 2 3\n";
        String expectedOutput = "1 2 3" + System.lineSeparator() +
                "0" + System.lineSeparator();

        // Act
        runAppWithInput(input);

        // Assert
        assertEquals(expectedOutput, outContent.toString());
    }

    /**
     * Tests the sample case provided in the assignment description.
     * Verifies the path and the final cost.
     * @throws Exception if the test run fails.
     */
    @Test
    void testFromAssignmentSample() throws Exception {
        // Arrange
        String input = "9 7 8\n7 8 9\n";
        String expectedOutput = "9 7 8" + System.lineSeparator() +
                "8 7 9" + System.lineSeparator() +
                "7 8 9" + System.lineSeparator() +
                "22" + System.lineSeparator();

        // Act
        runAppWithInput(input);

        // Assert
        assertEquals(expectedOutput, outContent.toString());
    }

    /**
     * Helper method to assert the final state and total cost from the captured output.
     * @param expectedState The expected string representation of the final array.
     * @param expectedCost The expected total integer cost.
     */
    private void assertFinalStateAndCost(String expectedState, int expectedCost) {
        String[] lines = outContent.toString().split(System.lineSeparator());
        String finalState = lines[lines.length - 2];
        int finalCost = Integer.parseInt(lines[lines.length - 1]);
        assertEquals(expectedState, finalState);
        assertEquals(expectedCost, finalCost);
    }

    /**
     * Tests if the algorithm correctly prefers a path with a cheap swap between even numbers
     * over more expensive initial moves.
     * @throws Exception if the test run fails.
     */
    @Test
    void testPreferCheapEvenSwap() throws Exception {
        // Arrange
        String input = "4 1 2\n1 2 4\n";
        runAppWithInput(input);
        assertFinalStateAndCost("1 2 4",13);
    }

    /**
     * Tests a scenario where expensive swaps between odd numbers are required.
     * @throws Exception if the test run fails.
     */
    @Test
    void testExpensiveOddSwap() throws Exception {
        String input = "3 5 1\n1 3 5\n";
        runAppWithInput(input);
        assertFinalStateAndCost("1 3 5", 40);
    }
    
    /**
     * Tests the solver with a slightly longer array to check scalability.
     * @throws Exception if the test run fails.
     */
    @Test
    void testLongerArray() throws Exception
    {
        String input = "4 3 2 1\n1 2 3 4\n";
        runAppWithInput(input);
        assertFinalStateAndCost("1 2 3 4", 22);
    }

    /**
     * Tests the solver's behavior with an array containing repeated numbers.
     * @throws Exception if the test run fails.
     */
    @Test
    void testWithRepeatedNumbers() throws Exception {
        // Arrange
        String input = "2 1 2\n1 2 2\n";
        String expectedOutput = "2 1 2" + System.lineSeparator() +
                "1 2 2" + System.lineSeparator() +
                "11" + System.lineSeparator();

        // Act
        runAppWithInput(input);

        // Assert
        assertEquals(expectedOutput, outContent.toString());
    }

    /**
     * Tests if the Uniform-Cost Search correctly avoids a "greedy trap".
     * It should choose a path with a higher initial step cost if it leads to a lower overall solution cost.
     * @throws Exception if the test run fails.
     */
    @Test
    void testOptimalPathOverGreedyTrap() throws Exception {
        // Arrange
        String input = "2 9 4 7\n7 9 4 2\n";
        // A "greedy" first move would be to swap 2 and 4 (cost 2), leading to state "4 9 2 7".
        // The total cost from there would be 2 (2<>4) + 11 (4<>7) + 2 (2<>4) = 15.
        // The optimal path is a single swap of 2 and 7 (cost 11).
        String expectedOutput = "2 9 4 7" + System.lineSeparator() +
                                "7 9 4 2" + System.lineSeparator() +
                                "11" + System.lineSeparator();

        // Act
        runAppWithInput(input);

        // Assert
        assertEquals(expectedOutput, outContent.toString());
    }

    /**
     * Tests the case where no solution is possible because the goal state is unreachable
     * (contains elements not present in the initial state).
     * @throws Exception if the test run fails.
     */
    @Test
    void testNoSolutionFound() throws Exception {
        // Arrange
        // The goal contains elements not present in the initial state.
        String input = "1 2 3\n4 5 6\n";
        String expectedOutput = "no solution found" + System.lineSeparator();

        // Act
        runAppWithInput(input);

        // Assert
        assertEquals(expectedOutput, outContent.toString());
    }

    /**
     * Tests the simplest non-trivial case: a two-element array that needs swapping.
     * @throws Exception if the test run fails.
     */
    @Test
    void testTwoElementArray() throws Exception {
        // Arrange
        String input = "10 1\n1 10\n";
        String expectedOutput = "10 1" + System.lineSeparator() +
                                "1 10" + System.lineSeparator() +
                                "11" + System.lineSeparator();

        // Act
        runAppWithInput(input);

        // Assert
        assertEquals(expectedOutput, outContent.toString());
    }

    /**
     * Tests a scenario with only even numbers, where all swaps should have a cost of 2.
     * @throws Exception if the test run fails.
     */
    @Test
    void testAllEvenNumbers() throws Exception {
        // Arrange
        String input = "8 6 4 2\n2 4 6 8\n";
        // All swaps should have a cost of 2.
        // A possible path: 8642 -> 2648 (c=2) -> 2468 (c=2). Total=4
        // Let's trace a more likely UCS path:
        // 8642 -> 2648 (c=2)
        // 2648 -> 2468 (c=2) -> Total 4
        runAppWithInput(input);
        assertFinalStateAndCost("2 4 6 8", 4);
    }

    /**
     * Tests a scenario with only odd numbers, where a single swap should cost 20.
     * @throws Exception if the test run fails.
     */
    @Test
    void testAllOddNumbers() throws Exception {
        // Arrange
        String input = "5 3 1\n1 3 5\n";
        // All swaps should have a cost of 20.
        runAppWithInput(input);
        assertFinalStateAndCost("1 3 5", 20);
    }

    /**
     * Tests the performance of the solver on a moderately complex problem.
     * The test will fail if it takes longer than 1 second to complete, ensuring
     * the algorithm is efficient enough for non-trivial cases.
     * This problem involves sorting a reversed array of 7 elements.
     * @throws Exception if the main method throws an exception.
     */
    @Test
    @Timeout(value = 100, unit = TimeUnit.MILLISECONDS)
    void testPerformanceOnModeratelyComplexProblem() throws Exception
    {
        // Arrange
        String input = "7 6 5 4 3 2 1\n1 2 3 4 5 6 7\n";

        // Act
        runAppWithInput(input);

        // Assert: We just need it to complete in time. A full assertion is good practice.
        assertFinalStateAndCost("1 2 3 4 5 6 7", 42);
    }

    /**
     * Provides an estimate of the memory usage for solving a complex problem.
     * This test measures the difference in used heap memory before and after the solver runs.
     * Note: This is an approximation, as the Garbage Collector's behavior can influence the results.
     * The output is printed to the console for observation.
     */
    @Test
    void testMemoryUsage() {
        // Arrange
        GSolver gs = new GSolver();
        Ilayout initial = new ArrayCfg("7 6 5 4 3 2 1");
        Ilayout goal = new ArrayCfg("1 2 3 4 5 6 7");

        // Suggest GC run to get a stable baseline
        System.gc();

        long memoryBefore = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();

        // Act
        Iterator<GSolver.State> it = gs.solve(initial, goal);

        long memoryAfter = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();

        // Assert & Report
        System.out.println("\n--- Memory Usage Test ---");
        System.out.printf("Estimated memory used by solver: %.3f MB%n", (memoryAfter - memoryBefore) / (1024.0 * 1024.0));
        System.out.println("-------------------------");
    }
}