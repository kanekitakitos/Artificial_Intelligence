
import core.*;

import java.util.Iterator;
import java.util.Scanner;

/**
 * Main class serving as the entry point for the 8-puzzle solver application.
 *
 * @preConditions:
 *                 - The user must provide two valid board configurations via standard input, separated by a space.
 *                 - Each configuration must be a string of 9 characters representing the board,
 *                   with '1' through '8' for the tiles and '.' for the empty space.
 *                 - Example input: 12345678. 1234567.8
 *
 * @postConditions:
 *                  - The application attempts to find the shortest path from the initial to the goal state.
 *                  - If a solution is found, each state (board configuration) in the solution path is printed to standard output.
 *                  - The final line of the output will be the total cost 'g' of the solution path.
 *                  - If no solution is found, the message "no solution found" is printed.
 *
 *                  This class orchestrates the puzzle-solving process by:
 *                  1. Reading the initial and goal board layouts from standard input.
 *                  2. Instantiating the `BestFirst` search algorithm.
 *                  3. Invoking the `solve` method with the initial and goal layouts.
 *                  4. Iterating through the solution path and printing each state.
 *
 * @see core.BestFirst
 * @see core.Board
 * @see core.Ilayout
 *
 * @author Brandon Mejia
 * @version 2025-09-07
 */
public class Main {
    /**
     * Main method that reads the initial and goal board configurations from standard input,
     * solves the puzzle using the BestFirst search algorithm, and prints the solution path.
     * 
     * @param args Command line arguments (not used)
     * @throws Exception If an error occurs during input processing
     */
    public static void main (String [] args) throws Exception {
        Scanner sc = new Scanner(System.in);

        BestFirst s = new BestFirst();
        Iterator<BestFirst.State> it = s.solve(new Board(sc.next()),
                new Board(sc.next()));
        if (it==null) System.out.println("no solution found");
        else {
            while(it.hasNext()) {
                BestFirst.State i = it.next();
                System.out.println(i);

                if (!it.hasNext()) System.out.println((int)i.getG());
            }
        }
        sc.close();
    }
}
