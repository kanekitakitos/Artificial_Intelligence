package core;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Represents an immutable state of the 8-puzzle problem.
 * <p>
 * This class implements the {@link Ilayout} interface, providing the problem-specific logic
 * for the 8-puzzle. It is designed to be immutable to ensure that states cannot be
 * accidentally modified during the search process.
 *
 * <h3>Key Implementation Details</h3>
 * <h4>Immutability</h4>
 * <p>
 * Immutability is achieved by making all fields {@code final} and by creating deep copies
 * of the board state when generating successors in the {@link #children()} method.
 * </p>
 *
 * <h4>State Representation</h4>
 * <p>
 * The board is represented as a 3x3 2D integer array. The empty space is represented
 * by the integer {@code 0}.
 * </p>
 *
 * @see Ilayout
 * @see AbstractSearch
 * @see BestFirst
 *
 * @author Brandon Mejia
 * @version 2025-09-30
 */
public class Board implements Ilayout {
    private static final int dim = 3;
    private final int[][] board;

    /**
     * Constructs a board from a compact string representation.
     * <p>
     * The input string must be exactly 9 characters long, containing the digits '1' through '8'
     * and a single '.' character to represent the empty space.
     * </p>
     *
     * @param str The string representing the board layout (e.g., "12345.786").
     * @throws IllegalArgumentException if the input string does not have a length of 9.
     */
    public Board(String str)
    {
        if (str.length() != dim * dim)
            throw new IllegalArgumentException("Invalid string length for Board constructor. Must be 9.");

        board = new int[dim][dim];
        int si = 0;

        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
                char c = str.charAt(si++);
                // Convert '.' to 0 for the empty space, otherwise parse the digit.
                board[i][j] = (c == '.') ? 0 : Character.getNumericValue(c);
            }
        }
    }

    /**
     * Private constructor for efficient cloning.
     * <p>
     * Creates a deep copy of an existing board state, used internally by the {@link #children()}
     * method to ensure immutability when creating new successor states.
     *
     * @param existingBoard The 2D array of an existing board to be cloned.
     */
    private Board(int[][] existingBoard) {
        this.board = new int[dim][dim];
        for (int i = 0; i < dim; i++) {
            this.board[i] = Arrays.copyOf(existingBoard[i], dim);
        }
    }

    /**
     * Returns a string representation of the board for display.
     * <p>
     * The empty space (represented internally as 0) is converted to a space character (' ')
     * for a more readable, traditional 8-puzzle format.
     *
     * @return A formatted, multi-line string of the board.
     */
    @Override
    public String toString()
    {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
                sb.append(board[i][j] == 0 ? " " : board[i][j]);
            }
            sb.append(System.lineSeparator());
        }
        return sb.toString();
    }

    /**
     * Gets the cost of a single move. For the standard 8-puzzle problem, every
     * move (sliding a tile) has a uniform cost of 1.
     *
     * @return The constant cost of a move (1.0).
     */
    @Override
    public double getK() {
        return 1.0;
    }

    @Override
    public double getH(Ilayout goal) {
        return 0;
    }

    /**
     * Checks if this board is the goal state by comparing it to another layout.
     * This method delegates the comparison to the {@link #equals(Object)} method, which
     * performs a deep comparison of the board states.
     *
     * @param l The goal layout to compare against.
     * @return `true` if this board is identical to the goal layout, `false` otherwise.
     */
    @Override
    public boolean isGoal(Ilayout l) {
        return this.equals(l);
    }

    /**
     * Generates all valid successor states (children) by moving the empty tile.
     * <p>
     * A successor is created for each valid move of the empty tile (up, down, left, or right)
     * into an adjacent position.
     *
     * @return A list of new `Board` objects representing all possible next states.
     */
    @Override
    public List<Ilayout> children()
    {
        List<Ilayout> children = new ArrayList<>();
        int zeroRow = -1, zeroCol = -1;

        // Find the position of the empty space (0)
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
                if (board[i][j] == 0) {
                    zeroRow = i;
                    zeroCol = j;
                    break;
                }
            }
        }

        // Define possible moves: Up, Down, Left, Right
        int[] dRow = {-1, 1, 0, 0};
        int[] dCol = {0, 0, -1, 1};

        for (int i = 0; i < 4; i++) {
            int newRow = zeroRow + dRow[i];
            int newCol = zeroCol + dCol[i];

            // If the new position is within the board boundaries
            if (newRow >= 0 && newRow < dim && newCol >= 0 && newCol < dim) {
                Board child = new Board(this.board); // Create a clone
                // Swap the empty tile with the adjacent tile
                child.board[zeroRow][zeroCol] = child.board[newRow][newCol];
                child.board[newRow][newCol] = 0;
                children.add(child);
            }
        }
        return children;
    }

    /**
     * Compares this board with another object for equality.
     * <p>
     * Two boards are considered equal if they are both of type {@code Board} and have
     * the exact same tile configuration in all positions.
     *
     * @param o The object to compare with.
     * @return `true` if the other object is a `Board` with the exact same tile configuration.
     */
    @Override
    public boolean equals(Object o)
    {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Board other = (Board) o;
        return Arrays.deepEquals(this.board, other.board);
    }

    /**
     * Computes the hash code for this board.
     * <p>
     * The hash code is based on the contents of the board's 2D array, ensuring that
     * two equal boards will have the same hash code, as required for collections like {@link java.util.HashMap}.
     * The hash code is based on the contents of the board's 2D array.
     *
     * @return The hash code for this board.
     */
    @Override
    public int hashCode() {
        return Arrays.deepHashCode(board);
    }
}
