package test;

import org.junit.jupiter.api.Test;
import java.io.PrintWriter;
import java.io.StringWriter;
import static org.junit.jupiter.api.Assertions.assertEquals;
import core.Board;

/**
 * Unit tests for the 8-puzzle implementation.
 * Tests the Board class functionality including construction and string representation.
 *
 * @author Brandon Mejia
 * @date 2025-09-07
 */
public class PuzzleUnitTests {

    /**
     * Tests the Board constructor and toString method with a specific board configuration.
     * Verifies that the string representation matches the expected format.
     */
    @Test
    public void testConstructorAndToString() {
        Board b = new Board("023145678");
        StringWriter writer = new StringWriter();
        PrintWriter pw = new PrintWriter(writer);
        pw.println(" 23");
        pw.println("145");
        pw.print("678"); // Use print para a última linha para evitar newline extra
        pw.close();

        // Normalizar quebras de linha para consistência entre sistemas operativos
        String expected = writer.toString().replaceAll("\\r\\n", "\n");
        String actual = b.toString().replaceAll("\\r\\n", "\n");
        assertEquals(expected, actual);
    }

    /**
     * Tests the Board constructor and toString method with an alternative board configuration.
     * Verifies that the string representation matches the expected format using system-specific line separators.
     */
    @Test
    public void testConstructorAndToString2() {
        Board b = new Board("123485670");
        String expected = "123" + System.lineSeparator() + "485" + System.lineSeparator() + "67 ";

        String actual = b.toString();
        assertEquals(expected, actual);
    }
}