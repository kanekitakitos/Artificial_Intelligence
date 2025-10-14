package test;

import core.ArrayCfg;
import core.AStarSearch;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.time.Duration;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.io.PrintStream;
import java.util.Iterator;
import java.util.Scanner;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTimeout;

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
    void testSample1_performance() {
        // Arrange
        String input = "-2 4 0 -1 3 5 1\n-2 -1 0 1 3 4 5\n";
        String expectedOutput = "33" + System.lineSeparator();

        // Act & Assert
        assertTimeout(Duration.ofSeconds(1), () -> {
            runAppWithInput(input);
            assertEquals(expectedOutput, outContent.toString(), "The output should be correct even when testing for performance.");
        });
    }

    /**
     * Tests the performance on a more complex case (a reversed array of 8 elements).
     * This ensures the heuristic is effective enough to solve larger problems quickly.
     * The timeout is set to 1 second.
     */
    @Test
    void testReversedArray_performance() {
        // Arrange
        String input = "8 7 6 5 4 3 2 1\n1 2 3 4 5 6 7 8\n";
        String expectedOutput = "44" + System.lineSeparator();

        // Act & Assert
        assertTimeout(Duration.ofSeconds(1), () -> {
            runAppWithInput(input);
            assertEquals(expectedOutput, outContent.toString(), "The output for the reversed array should be correct.");
        });
    }

    /**
     * A more demanding performance test with a 10-element array where even and odd numbers
     * are completely scrambled relative to their final positions. This forces the algorithm
     * to navigate a larger search space with varied step costs.
     * The timeout is kept at 1 second to ensure high performance.
     */
    @Test
    void testHighlyScrambled_10_elements_performance() {
        // Arrange: Evens are in odd positions and vice-versa.
        String input = "9 8 7 6 5 4 3 2 1 10\n1 2 3 4 5 6 7 8 9 10\n";
        // The optimal solution requires 4 swaps, all between even and odd numbers.
        // (9,1), (8,2), (7,3), (6,4). The 5 and 10 are in correct final positions relative to each other.
        // A better path is swapping pairs that are in each other's places: (10,1), (2,9), (4,7), (6,8). Cost: 4*11=44
        String expectedOutput = "44" + System.lineSeparator();

        // Act & Assert
        assertTimeout(Duration.ofSeconds(1), () -> {
            runAppWithInput(input);
            assertEquals(expectedOutput, outContent.toString(), "The output for the highly scrambled array should be correct.");
        });
    }
}