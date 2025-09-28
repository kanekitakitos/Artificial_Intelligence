package core;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Represents an 8-puzzle board, implementing the {@link Ilayout} interface.
 *
 * @preConditions
 *                 - The constructor must be called with a valid representation of a board.
 *
 * @postConditions
 *                  - An immutable `Board` object is created, representing a specific state of the 8-puzzle.
 *                  - The board can generate its children, be compared to a goal, and provide its step cost.
 *
 *                  This class defines the problem-specific logic for the 8-puzzle.
 *                  It stores the board as a 2D integer array, where `0` represents the empty space.
 *                  It provides the necessary methods for a search algorithm like `BestFirst` to navigate the state space.
 *
 * @see Ilayout
 * @see BestFirst
 *
 * @author Brandon Mejia
 * @version 2025-09-07
 */
public class Board implements Ilayout {
    private static final int dim = 3;
    private final int[][] board;

    /**
     * Constructs a board from a string representation.
     *
     * @preConditions
     *                 - The input string `str` must have exactly `dim * dim` (9) characters.
     *                 - The string must contain characters '1' through '8' and one '.' for the empty space.
     * @postConditions
     *                  - A new `Board` object is initialized with the specified tile configuration.
     *                  - The '.' character is converted to `0` to represent the empty space internally.
     *
     * @param str The string representing the board layout.
     * @throws IllegalStateException if the input string does not have a length of 9.
     */
    public Board(String str) throws IllegalStateException
    {
        if (str.length() != dim * dim)
            throw new IllegalStateException("Invalid arg in core.Board constructor");

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
     * Private constructor for efficient cloning. Creates a deep copy of an existing board.
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
     * The empty space (0) is represented by a space character.
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
            if (i < dim - 1) {
                sb.append(System.lineSeparator());
            }
        }
        return sb.toString() + '\n';
    }

    /**
     * Gets the cost of a single move, which is always 1.0 for the 8-puzzle problem.
     *
     * @return The constant cost of a move (1.0).
     */
    @Override
    public double getG() {
        return 1.0;
    }

    /**
     * Checks if this board is the goal state by comparing it to another layout.
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
     * The hash code is based on the contents of the board's 2D array.
     *
     * @return The hash code for this board.
     */
    @Override
    public int hashCode() {
        return Arrays.deepHashCode(board);
    }
}
