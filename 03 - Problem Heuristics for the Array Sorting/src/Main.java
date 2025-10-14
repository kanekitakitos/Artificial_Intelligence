
import core.ArrayCfg;
import core.BestFirst;
import core.Board;
import core.GSolver;
import java.util.Iterator;
import java.util.Scanner;

/**
 * Entry point for the search algorithm applications.
 * <p>
 * This class contains the main method that orchestrates the execution of different search problems.
 * It includes separate static methods to run the array sorting problem ({@link #GSolve()}) and
 * the 8-puzzle problem ({@link #bestFirst()}). A simple boolean flag controls which
 * problem is executed.
 *
 * @author Brandon Mejia
 * @version 2025-09-27
 */
public class Main {
    public static void main (String [] args)
    {
        boolean gSolve = true;

        if(gSolve)
            GSolve();
        else
            bestFirst();
    }

    /**
     * Executes the array sorting problem solver.
     * <p>
     * This method reads an initial and a goal configuration from standard input,
     * uses the {@link GSolver} (Uniform-Cost Search) to find the optimal solution,
     * and prints the total cost of the path to standard output.
     */
    public static void GSolve()
    {
        Scanner sc = new Scanner(System.in);
        GSolver gs = new GSolver();
        Iterator<GSolver.State> it =
                gs.solve( new ArrayCfg(sc.nextLine()), new ArrayCfg(sc.nextLine()));
        
        if (it==null) System.out.println("no solution found");
        else {
            // Itera sobre o caminho da solução para imprimir cada passo
            while(it.hasNext()) {
                GSolver.State i = it.next();
                //System.out.println(i); // Imprime o estado atual (o layout)

                // Se for o último estado, imprime o custo total
                if (!it.hasNext()) System.out.println((int)i.getK());
            }
        }
        sc.close();

    }
    /**
     * Executes the 8-puzzle problem solver.
     * <p>
     * This method reads an initial and a goal configuration for the 8-puzzle from standard input,
     * uses the {@link BestFirst} search algorithm to find a solution, and prints the
     * full path from the initial state to the goal state, followed by the total cost.
     */
    public static void bestFirst() {
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
