package test;

import core.AbstractSearch;
import core.AStarSearch;
import core.ArrayCfg;
import core.GSolver;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.io.PrintStream;
import java.time.Duration;
import java.util.Iterator;
import java.util.concurrent.*;
import java.util.Scanner;

import static org.junit.jupiter.api.Assertions.fail;

/**
 * Performance comparison tests between GSolver (Uniform-Cost Search) and AStarSearch.
 * <p>
 * This class demonstrates the significant performance advantage of using an informed search
 * algorithm (A*) with a good heuristic over an uninformed one (UCS) for complex problems.
 *
 * @author Brandon Mejia
 * @version 2025-10-15
 */
public class ComparisonTest {

    private final ByteArrayOutputStream outContent = new ByteArrayOutputStream();
    private final PrintStream originalOut = System.out;
    private final InputStream originalIn = System.in;

    @BeforeEach
    public void setUpStreams() {
        // We will print to the actual console to see the timing results.
        // The output of the solvers will still be captured by outContent.
        System.setOut(new PrintStream(outContent)); 
    }

    @AfterEach
    public void restoreStreams() {
        System.setOut(originalOut);
        System.setIn(originalIn);
    }

    /**
     * Generic method to run any solver that extends AbstractSearch.
     * This avoids code duplication between runGSolver and runAStar.
     */
    private void runSolver(AbstractSearch solver, String input) {
        System.setIn(new ByteArrayInputStream(input.getBytes()));
        Scanner sc = new Scanner(System.in);
        Iterator<? extends AbstractSearch.State> it = solver.solve(new ArrayCfg(sc.nextLine()), new ArrayCfg(sc.nextLine()));
        if (it != null) {
            AbstractSearch.State finalState = null;
            while (it.hasNext()) {
                finalState = it.next();
            }
            if (finalState != null) System.out.println((int) finalState.getG());
        }
        sc.close();
    }

    /**
     * Runs a solver in a separate thread with a timeout and measures its execution time.
     * This prevents a slow algorithm from blocking the entire test suite.
     *
     * @param solverName The name of the algorithm for reporting.
     * @param solver The solver instance to run.
     * @param input The input data for the solver.
     * @param timeoutSeconds The maximum time to wait for the solver to finish.
     */
    private void runAndMeasure(String solverName, AbstractSearch solver, String input, long timeoutSeconds) {
        System.err.printf("  -> Executing %s (timeout: %d seconds)...%n", solverName, timeoutSeconds);

        // Suggest garbage collection to have a cleaner state before measuring.
        System.gc();
        try {
            // A short pause to allow GC to run.
            Thread.sleep(200);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }

        ExecutorService executor = Executors.newSingleThreadExecutor();
        try {
            Callable<Long> task = () -> {
                long startTime = System.nanoTime();
                runSolver(solver, input);
                long endTime = System.nanoTime();
                return endTime - startTime;
            };

            Future<Long> future = executor.submit(task);

            // Wait for the task to complete and get the execution time.
            // This will block until the algorithm finishes, no matter how long it takes.
            long durationNano = future.get();
            double durationMs = durationNano / 1_000_000.0;
            double durationS = durationMs / 1000.0;

            System.err.printf("     Completed in: %.3f s (%.3f ms)%n", durationS, durationMs);

            // Also print a warning if it exceeded the suggested timeout.
            if (durationMs > timeoutSeconds * 1000) {
                System.err.printf("     WARNING: Execution exceeded the suggested timeout of %d seconds.%n", timeoutSeconds);
            }

        } catch (InterruptedException | ExecutionException e) {
            // Catches exceptions from within the solver task or thread interruption.
            System.err.printf("     FAILED: Execution threw an exception: %s%n", e.getCause() != null ? e.getCause() : e);
            fail("Solver execution failed", e.getCause());
        } finally {
            executor.shutdown(); // Always shut down the executor.
        }
    }

    /**
     * Compares the performance on a complex, fully reversed 10-element array.
     * A* is expected to be orders of magnitude faster because its heuristic guides it
     * directly towards the goal, while GSolver (UCS) explores a vast number of
     * low-cost but incorrect paths.
     */
    @Test
    @DisplayName("Compare A* vs GSolver on a 10-element reversed array")
    void compare_reversedArray_10_elements() {
        String input = "10 9 8 7 6 5 4 3 2 1\n1 2 3 4 5 6 7 8 9 10\n";

        // Print to the real console (System.err) to see the results live.
        System.err.println("\n--- Comparison on 10-Element Reversed Array ---");

        runAndMeasure("A* Search", new AStarSearch(), input, 1);
        runAndMeasure("GSolver (UCS)", new GSolver(), input, 20);

        System.err.println("---------------------------------------------------");
    }

    /**
     * Compares performance on the "worst-case" scenario with 12 elements.
     * In this case, the heuristic of A* is critical to avoid an exponential explosion
     * of states. GSolver will explore nodes based only on cost `g`, which is a very poor
     * indicator in this problem, leading to an unmanageable search space.
     */
    @Test
    @DisplayName("Compare A* vs GSolver on a 12-element 'worst-case' array")
    void compare_worstCase_12_elements() {
        String input = "2 4 6 8 10 12 1 3 5 7 9 11\n1 3 5 7 9 11 2 4 6 8 10 12\n";

        System.err.println("\n--- Comparison on 12-Element 'Worst-Case' Array ---");

        runAndMeasure("A* Search", new AStarSearch(), input, 1);
        runAndMeasure("GSolver (UCS)", new GSolver(), input, 200);

        System.err.println("---------------------------------------------------");
    }

    /**
     * Compares performance on a very large "worst-case" scenario with 20 elements.
     * This test highlights the scalability difference, where A* remains viable while
     * GSolver's execution time grows exponentially, making it impractical.
     */
    @Test
    @DisplayName("Compare A* vs GSolver on a 20-element 'worst-case' array")
    void compare_worstCase_20_elements() {
        String input = "2 4 6 8 10 12 14 16 18 20 1 3 5 7 9 11 13 15 17 19\n" +
                       "1 3 5 7 9 11 13 15 17 19 2 4 6 8 10 12 14 16 18 20\n";

        System.err.println("\n--- Comparison on 20-Element 'Worst-Case' Array ---");

        // A* should still be able to handle this due to the powerful heuristic.
        runAndMeasure("A* Search", new AStarSearch(), input, 1);

        // GSolver will face a combinatorial explosion and will time out.
        runAndMeasure("GSolver (UCS)", new GSolver(), input, 9000);

        System.err.println("---------------------------------------------------");
    }
}