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



    private void main (String [] args)
    {
        Scanner sc = new Scanner(System.in);
        GSolver gs = new GSolver();
        Iterator<GSolver.State> it =
                gs.solve( new ArrayCfg(sc.nextLine()), new ArrayCfg(sc.nextLine()));
        if (it==null) System.out.println("no solution found");
        else {
            while(it.hasNext()) {
                GSolver.State i = it.next();
                System.out.println(i);
                if (!it.hasNext()) System.out.println((int)i.getK());
            }
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
     * @throws Exception if the main method throws an exception.
     */
    private void runAppWithInput(String input) throws Exception {
        ByteArrayInputStream inContent = new ByteArrayInputStream(input.getBytes());
        System.setIn(inContent);
        // Call the actual main method from the project's Main class
        main(null);
    }

    /**
     * Tests Sample 1 from the assignment description.
     * Verifies the path and the final cost.
     * @throws Exception if the test run fails.
     */
    @Test
    void testSample1() throws Exception {
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
     * Tests Sample 2 from the assignment description.
     * Verifies the path and the final cost.
     * @throws Exception if the test run fails.
     */
    @Test
    void testSample2() throws Exception {
        // Arrange
        String input = "6 8 2 5 10\n8 10 2 5 6\n";
        String expectedOutput = "6 8 2 5 10" + System.lineSeparator() +
                                "10 8 2 5 6" + System.lineSeparator() +
                                "8 10 2 5 6" + System.lineSeparator() +
                                "4" + System.lineSeparator();

        // Act
        runAppWithInput(input);

        // Assert
        assertEquals(expectedOutput, outContent.toString());
    }

    /**
     * Tests Sample 3 from the assignment, which has multiple optimal paths.
     * @throws Exception if the test run fails.
     */
    @Test
    void testSample3() throws Exception {
        // Arrange
        String input = "14 11 15 13 12\n15 14 13 12 11\n";
        String expectedOutput = "14 11 15 13 12" + System.lineSeparator() +
                                "14 12 15 13 11" + System.lineSeparator() +
                                "12 14 15 13 11" + System.lineSeparator() +
                                "15 14 12 13 11" + System.lineSeparator() +
                                "15 14 13 12 11" + System.lineSeparator() +
                                "35" + System.lineSeparator();
        // Act
        runAppWithInput(input);

        // Assert
        assertEquals(expectedOutput, outContent.toString());
    }
}