package test;

import core.ArrayCfg;
import core.GSolver;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.io.PrintStream;
import java.util.Iterator;
import java.util.Scanner;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Unit and integration tests for the {@link GSolver} class.
 * <p>
 * This class tests the complete application flow by simulating the execution of the main program.
 * It provides various test cases based on the assignment description and other edge cases.
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
     * A private replica of the main application logic for testing purposes.
     * This method reads from {@link System#in}, invokes the solver, and prints the
     * final cost to {@link System#out}.
     */
    private void main (String [] args)
    {
        Scanner sc = new Scanner(System.in);
        GSolver gs = new GSolver();
        Iterator<GSolver.State> it =
                gs.solve( new ArrayCfg(sc.nextLine()), new ArrayCfg(sc.nextLine()));
        if (it==null) System.out.println("no solution found");
        else {
            // Iterate through the solution path to find the final state
            GSolver.State finalState = null;
            while(it.hasNext()) {
                finalState = it.next();
            }
            // Print only the cost of the final state
            if (finalState != null) System.out.println((int)finalState.getG());
        }
        sc.close();
    }


    /**
     * Before each test, redirect System.out to our stream so we can capture the output.
     */
    @BeforeEach
    public void setUpStreams() {
        System.setOut(new PrintStream(outContent));
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
     */
    private void runAppWithInput(String input) throws Exception {
        ByteArrayInputStream inContent = new ByteArrayInputStream(input.getBytes());
        System.setIn(inContent);
        // Call the actual main method from the project's Main class
        main(null);
    }

    /**
     * Tests Sample 1 from the assignment description. Verifies the final cost.
     * @throws Exception if the test run fails.
     */
    @Test
    void testSample1() throws Exception {
        // Arrange
        String input = "9 7 8\n7 8 9\n";
        String expectedOutput = "22" + System.lineSeparator();

        // Act
        runAppWithInput(input);

        // Assert
        assertEquals(expectedOutput, outContent.toString());
    }
    
    /**
     * Tests Sample 2 from the assignment description. Verifies the final cost.
     * @throws Exception if the test run fails.
     */
    @Test
    void testSample2() throws Exception {
        // Arrange
        String input = "6 8 2 5 10\n8 10 2 5 6\n";
        String expectedOutput = "4" + System.lineSeparator();

        // Act
        runAppWithInput(input);

        // Assert
        assertEquals(expectedOutput, outContent.toString());
    }

    /**
     * Tests Sample 3 from the assignment. Verifies the final cost.
     * @throws Exception if the test run fails.
     */
    @Test
    void testSample3() throws Exception {
        // Arrange
        String input = "14 11 15 13 12\n15 14 13 12 11\n";
        String expectedOutput = "35" + System.lineSeparator();
        // Act
        runAppWithInput(input);

        // Assert
        assertEquals(expectedOutput, outContent.toString());
    }

    /**
     * Tests a specific case provided by the user, which is identical to Sample 2.
     * Verifies the final cost.
     * @throws Exception if the test run fails.
     */
    @Test
    void test04() throws Exception {
        // Arrange
        String input = "6 8 2 5 10\n8 10 2 5 6\n";
        String expectedOutput = "4" + System.lineSeparator();

        // Act
        runAppWithInput(input);

        // Assert
        assertEquals(expectedOutput, outContent.toString());
    }
    /**
     * Tests the algorithm's robustness with negative numbers and zero.
     * This case verifies that the parity checks and cost calculations are correct
     * for all integers, leading to the optimal path cost of 31.
     * @throws Exception if the test run fails.
     */
    @Test
    void test05() throws Exception {
        // Arrange
        String input = "-2 0 -1 -3 1\n-2 -1 0 1 -3\n";
        String expectedOutput = "31" + System.lineSeparator();

        // Act
        runAppWithInput(input);

        // Assert
        assertEquals(expectedOutput, outContent.toString());
    }

    /**
     * Tests the case where the initial state is already the goal state.
     * The expected cost should be 0.
     * @throws Exception if the test run fails.
     */
    @Test
    void test06() throws Exception {
        // Arrange
        String input = "1 2 3 4 5\n1 2 3 4 5\n";
        String expectedOutput = "0" + System.lineSeparator();

        // Act
        runAppWithInput(input);

        // Assert
        assertEquals(expectedOutput, outContent.toString());
    }

    /**
     * Tests a more complex sorting scenario.
     * Verifies the final cost.
     * @throws Exception if the test run fails.
     */
    @Test
    void test07() throws Exception {
        // Arrange
        String input = "3 5 4 2 1\n1 2 3 4 5\n";
        String expectedOutput = "35" + System.lineSeparator();

        // Act
        runAppWithInput(input);

        // Assert
        assertEquals(expectedOutput, outContent.toString());
    }
}