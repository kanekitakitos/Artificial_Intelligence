package test;

import core.BestFirst;
import core.Board;
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
 * Unit and integration tests for the 8-puzzle problem solver using Best-First search.
 * This class tests the complete application flow by simulating the `Main` class's `bestFirst` method.
 * It provides test cases to validate the pathfinding and cost calculation for the 8-puzzle.
 * @author Brandon Mejia
 * @version 2025-09-27
 */
public class BestFirstTest {

    // To capture System.out
    private final ByteArrayOutputStream outContent = new ByteArrayOutputStream();
    private final PrintStream originalOut = System.out;

    // To provide System.in
    private final InputStream originalIn = System.in;

    /**
     * Private main method that replicates the logic from Main.bestFirst() for testing purposes.
     * @param args Command-line arguments (not used).
     */
    private void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        BestFirst s = new BestFirst();
        Iterator<BestFirst.State> it = s.solve(new Board(sc.next()),
                new Board(sc.next()));
        if (it == null) System.out.println("no solution found");
        else {
            while (it.hasNext()) {
                BestFirst.State i = it.next();
                System.out.println(i);

                if (!it.hasNext()) System.out.println((int) i.getG());
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
     */
    private void runAppWithInput(String input) {
        ByteArrayInputStream inContent = new ByteArrayInputStream(input.getBytes());
        System.setIn(inContent);
        main(null);
    }

    /**
     * Test case 1: The initial state is the same as the goal state.
     * The expected output is the initial board configuration and a cost of 0.
     */
    @Test
    void testGoalIsInitialState() {
        // Arrange
        String initial = "023147685";
        String goal = "023147685";
        String input = initial + "\n" + goal + "\n";

        // The Board.toString() method represents '0' as a space ' ' and adds two newlines at the end.
        String expectedBoard = " 23" + System.lineSeparator() + "147" + System.lineSeparator() + "685" + System.lineSeparator() + System.lineSeparator();
        String expectedCost = "0" + System.lineSeparator();
        String expectedOutput = expectedBoard + expectedCost;

        // Act
        runAppWithInput(input);

        // Assert
        assertEquals(expectedOutput.replace("\r\n", "\n"), outContent.toString().replace("\r\n", "\n"));
    }

    /**
     * Test case 2: A more complex path requiring multiple moves.
     * The expected output is a sequence of 7 board states and a final cost of 6.
     */
    @Test
    void test02() {
        // Arrange
        String initial = "023147685";
        String goal = "123405678";
        String input = initial + "\n" + goal + "\n";

        String s = System.lineSeparator();
        String nl = "\n"; // The Board.toString() adds a single newline character

        String board1 = " 23" + s + "147" + s + "685" + nl;
        String board2 = "123" + s + " 47" + s + "685" + nl;
        String board3 = "123" + s + "4 7" + s + "685" + nl;
        String board4 = "123" + s + "47 " + s + "685" + nl;
        String board5 = "123" + s + "475" + s + "68 " + nl;
        String board6 = "123" + s + "475" + s + "6 8" + nl;
        String board7 = "123" + s + "4 5" + s + "678" + nl;

        String expectedOutput =
                board1 + s +
                board2 + s +
                board3 + s +
                board4 + s +
                board5 + s +
                board6 + s +
                board7 + s +
                "6" + s;

        // Act
        runAppWithInput(input);

        // Assert
        assertEquals(expectedOutput.replace("\r\n", "\n"), outContent.toString().replace("\r\n", "\n"));
    }

    /**
     * Test case 3: A longer path to the goal.
     * The expected output is a sequence of 10 board states and a final cost of 9.
     */
    @Test
    void test03() {
        // Arrange
        String initial = "216408753";
        String goal = "281430765";
        String input = initial + "\n" + goal + "\n";

        String s = System.lineSeparator();
        String nl = "\n"; // The Board.toString() adds a single newline character

        String board01 = "216" + s + "4 8" + s + "753" + nl;
        String board02 = "216" + s + "48 " + s + "753" + nl;
        String board03 = "21 " + s + "486" + s + "753" + nl;
        String board04 = "2 1" + s + "486" + s + "753" + nl;
        String board05 = "281" + s + "4 6" + s + "753" + nl;
        String board06 = "281" + s + "46 " + s + "753" + nl;
        String board07 = "281" + s + "463" + s + "75 " + nl;
        String board08 = "281" + s + "463" + s + "7 5" + nl;
        String board09 = "281" + s + "4 3" + s + "765" + nl;
        String board10 = "281" + s + "43 " + s + "765" + nl;

        String expectedOutput =
                board01 + s +
                board02 + s +
                board03 + s +
                board04 + s +
                board05 + s +
                board06 + s +
                board07 + s +
                board08 + s +
                board09 + s +
                board10 + s +
                "9" + s;

        // Act
        runAppWithInput(input);

        // Assert
        assertEquals(expectedOutput.replace("\r\n", "\n"), outContent.toString().replace("\r\n", "\n"));
    }

    /**
     * Test case 4: Another pathfinding scenario.
     * The expected output is a sequence of 6 board states and a final cost of 5.
     */
    @Test
    void test04() {
        // Arrange
        String initial = "283164705";
        String goal = "283156740";
        String input = initial + "\n" + goal + "\n";

        String s = System.lineSeparator();
        String nl = "\n"; // The Board.toString() adds a single newline character

        String board1 = "283" + s + "164" + s + "7 5" + nl;
        String board2 = "283" + s + "164" + s + "75 " + nl;
        String board3 = "283" + s + "16 " + s + "754" + nl;
        String board4 = "283" + s + "1 6" + s + "754" + nl;
        String board5 = "283" + s + "156" + s + "7 4" + nl;
        String board6 = "283" + s + "156" + s + "74 " + nl;

        String expectedOutput =
                board1 + s +
                board2 + s +
                board3 + s +
                board4 + s +
                board5 + s +
                board6 + s +
                "5" + s;

        // Act
        runAppWithInput(input);

        // Assert
        assertEquals(expectedOutput.replace("\r\n", "\n"), outContent.toString().replace("\r\n", "\n"));
    }

    /**
     * Test case 6: A long and complex pathfinding scenario.
     * The expected output is a sequence of 13 board states and a final cost of 12.
     */
    @Test
    void test06() {
        // Arrange
        String initial = "123456780";
        String goal = "436718520";
        String input = initial + "\n" + goal + "\n";

        String s = System.lineSeparator();
        String nl = "\n"; // The Board.toString() adds a single newline character

        String board01 = "123" + s + "456" + s + "78 " + nl;
        String board02 = "123" + s + "456" + s + "7 8" + nl;
        String board03 = "123" + s + "4 6" + s + "758" + nl;
        String board04 = "1 3" + s + "426" + s + "758" + nl;
        String board05 = " 13" + s + "426" + s + "758" + nl;
        String board06 = "413" + s + " 26" + s + "758" + nl;
        String board07 = "413" + s + "726" + s + " 58" + nl;
        String board08 = "413" + s + "726" + s + "5 8" + nl;
        String board09 = "413" + s + "7 6" + s + "528" + nl;
        String board10 = "4 3" + s + "716" + s + "528" + nl;
        String board11 = "43 " + s + "716" + s + "528" + nl;
        String board12 = "436" + s + "71 " + s + "528" + nl;
        String board13 = "436" + s + "718" + s + "52 " + nl;

        String expectedOutput =
                board01 + s +
                board02 + s +
                board03 + s +
                board04 + s +
                board05 + s +
                board06 + s +
                board07 + s +
                board08 + s +
                board09 + s +
                board10 + s +
                board11 + s +
                board12 + s +
                board13 + s +
                "12" + s;

        // Act
        runAppWithInput(input);

        // Assert
        assertEquals(expectedOutput.replace("\r\n", "\n"), outContent.toString().replace("\r\n", "\n"));
    }
}