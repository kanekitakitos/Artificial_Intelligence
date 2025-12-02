package apps;
import neural.activation.IDifferentiableFunction;
import neural.activation.*;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.*;

import math.Matrix;

/**
 * Orchestrates a resilient and automated search for optimal MLP model hyperparameters through a parallelized grid search.
 * <p>
 * This class implements a Grid Search strategy to systematically explore multiple
 * combinations of hyperparameters, including learning rate, momentum, network topology,
 * and activation functions. It is designed for efficiency and resilience, making it
 * ideal for long-running tuning tasks that are prone to interruption.
 * </p>
 * <ul>
 *   <li><b>GPU Acceleration:</b> It offloads the intensive training tasks to the {@link GpuMLP23}
 *       trainer, which uses the ND4J library to leverage GPU computation.</li>
 *   <li><b>Parallel Execution:</b> By using a thread pool, it evaluates multiple model
 *       configurations concurrently, significantly reducing search time.</li>
 *   <li><b>Fault Tolerance:</b> The results of each trial are immediately saved to a log file. If the
 *       process is interrupted, it can be restarted and will automatically skip previously
 *       completed combinations, resuming from where it left off.</li>
 * </p>
 * <p>
 * The results of all trials are collected, sorted by accuracy, and presented in a
 * final summary report that highlights the best-performing combination.
 * </p>
 *
 * <h3>How It Works (Grid Search)</h3>
 * <p>
 * The {@link #runGridSearch()} method orchestrates the entire process. It first reads the
 * {@code tuning_results.log} file to identify which hyperparameter combinations have already
 * been evaluated. It then generates a list of all possible combinations and submits the
 * remaining ones as parallel tasks to an {@link ExecutorService}. As each task completes,
 * its result is immediately written to the log file to ensure no work is lost. This makes the
 * process restartable and resilient to failures.
 * </p>
 *
 * <h4>Example Usage</h4>
 * <p>
 * To start the hyperparameter tuning process, simply create an instance of this class
 * and invoke the {@link #runGridSearch()} method.
 * </p>
 * <ul>
 *   <li><b>Implementation:</b>
 *      <pre>{@code
public static void main(String[] args) {
    HyperparameterTuner tuner = new HyperparameterTuner();
    tuner.runGridSearch();
}
 *      }</pre>
 *   </li>
 * </ul>
 *
 * <h3>Troubleshooting and Known Issues</h3>
 * <h4>Intermittent `ArrayIndexOutOfBoundsException`</h4>
 * <p>
 * The console log may show intermittent `ArrayIndexOutOfBoundsException` failures, often with a
 * `null` error message. This is a known issue caused by a **race condition** when multiple
 * threads attempt to read the same data files concurrently. While the `try-catch` block
 * prevents the application from crashing, these failed trials represent wasted computation.
 * </p>
 * <p>
 * **Solution:** If these errors are frequent, the most effective solution is to reduce the level of
 * parallelism by setting `numThreads` to `1`. This will force the trials to run sequentially,
 * eliminating the file access conflict at the cost of longer execution time.
 * </p>
 *
 * <h3>Future Improvements</h3>
 * <p>
 * While Grid Search is exhaustive, it can be computationally expensive. For a more
 * efficient search in a large hyperparameter space, consider implementing:
 * </p>
 * <ul>
 *   <li><b>Random Search:</b> Instead of testing all combinations, Random Search samples a fixed
 *       number of random combinations. It often finds better models in less time.</li>
 *   <li><b>Bayesian Optimization:</b> An even more advanced technique that uses the results from
 *       previous trials to inform which combination to try next.</li>
 * </ul>
 *
 * @see GpuMLP23
 * @see ExecutorService
 * @see CompletionService
 * @author Brandon Mejia
 * @version 2025-11-30
 */
public class HyperparameterTuner {

    /**
     * O ficheiro onde os resultados da otimização são guardados.
     */
    private static final String RESULTS_FILE = "src/data/tuning_results.log";

    /**
     * Define se o treino será executado em GPU.
     * Se {@code true}, o número de threads será limitado a 1 para evitar sobrecarga de VRAM.
     * Se {@code false}, usará todos os núcleos da CPU para paralelismo máximo.
     */
    private static final boolean USE_GPU = true;

    private final int SEED = 8;

    // --- Hiperparámetros para a busca ---
    private final double[] learningRates = {0.01//,0.009,0.011,
                                        //0.005, 0.001,0.002,0.0225,0.0221,0.003
                                        //,0.0005,0.0001,0.0002
    };

    private final double[] momentums = {0.6//,0.65, 0.5, 0.55
    };

    private final int[][] topologies = {
            {400, 1, 1},
            //{400, 2, 1},
            //{400, 4, 1}

    };
    private final IDifferentiableFunction[][] activationFunctions = {
            {new Sigmoid(), new Sigmoid()},
            //{new TanH(), new TanH()},
            //{new ReLU(), new Sigmoid()} // ReLU para camadas ocultas, Sigmoid para a saída
    };

    /**
     * A simple data class to store the results of a single training trial.
     */
    private static class TuningResult implements Comparable<TuningResult> {
        final String paramsDescription;
        final double accuracy;

        TuningResult(String paramsDescription, double accuracy) {
            this.paramsDescription = paramsDescription;
            this.accuracy = accuracy;
        }

        @Override
        public int compareTo(TuningResult other) {
            // Ordena por acurácia em ordem decrescente (maior é melhor)
            return Double.compare(other.accuracy, this.accuracy);
        }

        @Override
        public String toString() {
            return String.format("Parameters: [%s] -> Accuracy: %.2f%%", paramsDescription, accuracy);
        }
    }

    /**
     * Executes a parallel grid search to find the best hyperparameter combination.
     * <p>
     * This method creates a list of all possible hyperparameter combinations and
     * uses a fixed-size thread pool to train and evaluate them in parallel.
     * Once all trials are complete, it prints a sorted summary of the results.
     * </p>
     */
    public void runGridSearch() {
        // Carrega as combinações já concluídas para evitar trabalho duplicado.
        Set<String> completedTrials = loadCompletedTrials();
        System.out.printf("Found %d previously completed trial(s). They will be skipped.\n", completedTrials.size());

        // --- Otimização: Carregar os dados UMA VEZ antes de iniciar a busca ---
        System.out.println("Pre-loading and caching datasets to optimize parallel trials...");
        DataHandler dataHandler = new DataHandler(SEED); // Usar uma seed fixa
        Matrix trainInputs = dataHandler.getTrainInputs();
        Matrix trainOutputs = dataHandler.getTrainOutputs();
        Matrix[] testData = DataHandler.loadDefaultTestData();
        System.out.printf("-> Datasets loaded: %d training samples, %d test samples.\n\n",
                dataHandler.getTrainingDataSize(),
                testData[0].rows());

        // Create all combinations of parameters to be tested.
        List<Callable<TuningResult>> tasks = new ArrayList<>();
        for (int[] topology : topologies) {
            for (IDifferentiableFunction[] functions : activationFunctions) {
                // Ensure the topology and activation function combination is valid.
                if (topology.length - 1 != functions.length) continue;
                for (double lr : learningRates) {
                    for (double momentum : momentums) {
                        String paramsDescription = String.format(
                                "Topology: %s, Functions: %s, LR: %.4f, Momentum: %.2f",
                                Arrays.toString(topology), getFunctionNames(functions), lr, momentum
                        );

                        // Se a combinação já foi testada, salta-a.
                        if (completedTrials.contains(paramsDescription)) {
                            continue;
                        }

                        // Create a task for the current combination.
                        tasks.add(() -> runTrial(paramsDescription, topology, functions, lr, momentum, trainInputs, trainOutputs, testData));
                    }
                }
            }
        }

        // --- Configuração do Paralelismo (CPU vs GPU) ---
        final int numThreads;
        final String mode;
        if (USE_GPU) {
            // Força a execução em série para garantir que cada tarefa tenha acesso exclusivo à VRAM da GPU.
            // Correr várias tarefas de GPU em paralelo quase sempre causa erros de OutOfMemory.
            numThreads = 1;
            mode = "GPU (Foco Total, Serializado)";
            System.out.println("Modo GPU ativado. As tarefas serão executadas uma a uma para maximizar o uso da GPU.");
        } else {
            numThreads = Runtime.getRuntime().availableProcessors(); // Usa todos os núcleos da CPU.
            mode = String.format("CPU (Paralelismo Máximo em %d núcleos)", numThreads);
            System.out.println("Modo CPU ativado. As tarefas serão distribuídas por todos os núcleos da CPU.");
        }

        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        // CompletionService decouples task submission from result retrieval, which is more efficient.
        CompletionService<TuningResult> completionService = new ExecutorCompletionService<>(executor);
        List<TuningResult> results = new ArrayList<>();

        System.out.printf("--- Iniciando a Busca de Hiperparâmetros com %d thread(s) | Modo: %s ---\n", numThreads, mode);
        System.out.printf("Total new combinations to test: %d\n\n", tasks.size());

        // Submit all tasks for execution.
        for (Callable<TuningResult> task : tasks) {
            completionService.submit(task);
        }

        try {
            // Retrieve results as they become available.
            for (int i = 0; i < tasks.size(); i++)
            {
                try {
                    // Wait for the next completed task, with a generous timeout.
                    Future<TuningResult> future = completionService.poll(20, TimeUnit.MINUTES);
                    if (future == null) {
                        System.err.printf("\n[WARNING] A training trial timed out after 20 minutes and was cancelled. Skipping to the next one.\n");
                        continue; // Move to the next result
                    }
                    TuningResult result = future.get();
                    results.add(result);
                    // Guarda o resultado imediatamente para garantir a persistência.
                    saveResult(result);
                    System.out.printf(">> Completed trial %d/%d. Result saved.\n", (i + 1), tasks.size());

                } catch (CancellationException e) {
                    System.err.println("A training trial was cancelled, possibly due to a timeout.");
                } catch (ExecutionException e) {
                    System.err.println("A training trial failed: " + e.getCause().getMessage());
                    e.getCause().printStackTrace();
                }
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            System.err.println("Hyperparameter search was interrupted while waiting for a result.");
        } finally {
            executor.shutdown();
        }

        // Recarrega todos os resultados (antigos e novos) para o relatório final
        results.clear();
        try (BufferedReader reader = new BufferedReader(new FileReader(RESULTS_FILE))) {
            String line;
            while ((line = reader.readLine()) != null) {
                results.add(parseResultFromString(line));
            }
        } catch (IOException e) {
            System.err.println("Could not reload results for final summary.");
        }

        // Sort results by validation error (ascending).
        Collections.sort(results);

        // --- Print Summary ---
        System.out.println("\n\n--- === Search Summary === ---");
        results.forEach(System.out::println);

        if (!results.isEmpty()) {
            TuningResult bestResult = results.get(0);
            System.out.println("\n--- BEST COMBINATION FOUND ---");
            System.out.printf("Parameters: %s\n", bestResult.paramsDescription);
            System.out.printf("Best Accuracy: %.2f%%\n", bestResult.accuracy);
        } else {
            System.out.println("\n--- No successful trials were completed. ---");
        }
        System.out.println("--- ======================== ---");
    }

    /**
     * Runs a single training and validation trial for a given set of hyperparameters.
     * <p>
     * This method orchestrates one complete experiment by:
     * <ol>
     *   <li>Instantiating a {@link GpuMLP23} trainer with the specified configuration.</li>
     *   <li>Executing the training process using the full training dataset.</li>
     *   <li>Evaluating the trained model against the test dataset to calculate its accuracy.</li>
     * </ol>
     *
     * @return A {@link TuningResult} object containing the parameters and final validation error.
     */
    private TuningResult runTrial(String paramsDescription, int[] topology, IDifferentiableFunction[] functions, double lr, double momentum, Matrix trainInputs, Matrix trainOutputs, Matrix[] testData) {
        try {
            System.out.println("--- Testing combination: " + paramsDescription + " ---");

            // --- GPU-ACCELERATED TRIAL ---
            // We use the new GpuMLP23 trainer, which leverages ND4J for GPU computation.
            GpuMLP23 gpuTrainer = new GpuMLP23(topology, functions, lr, momentum, 50000); // Using fewer epochs as GPU training is much faster per epoch
            gpuTrainer.train(trainInputs, trainOutputs);

            // --- TEST THE TRAINED NETWORK ---
            // After training, evaluate the model's accuracy on the test dataset.
            double accuracy = gpuTrainer.test(testData[0], testData[1]);
            System.out.printf("--- Finished GPU trial: [%s] -> Accuracy: %.2f%% ---\n", paramsDescription, accuracy);

            return new TuningResult(paramsDescription, accuracy);
        } catch (ArrayIndexOutOfBoundsException e) {
            // Use e.toString() for a more descriptive message, as e.getMessage() can be null.
            System.err.printf("--- FAILED trial: [%s] -> Exception: %s. Likely a data loading issue. ---\n", paramsDescription, e.toString());
            // Return a result with 0 accuracy to mark it as a failed trial.
            return new TuningResult(paramsDescription, 0.0);
        }
    }

    /**
     * Loads the descriptions of already completed trials from the results file.
     * @return A Set of strings, where each string describes a completed parameter combination.
     */
    private Set<String> loadCompletedTrials() {
        Set<String> completed = new HashSet<>();
        try (BufferedReader reader = new BufferedReader(new FileReader(RESULTS_FILE))) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (line.startsWith("Parameters: [")) {
                    try {
                        String params = line.substring(line.indexOf('[') + 1, line.indexOf(']'));
                        completed.add(params);
                    } catch (StringIndexOutOfBoundsException e)
                    {
                        // Ignora linhas mal formatadas.
                    }
                }
            }
        } catch (IOException e) {
            // O ficheiro pode não existir na primeira execução, o que é normal.
            System.out.println("Results log not found. Starting a new search.");
        }
        return completed;
    }

    /**
     * Appends a single trial result to the log file in a thread-safe manner.
     * @param result The TuningResult to save.
     */
    private synchronized void saveResult(TuningResult result) {
        try (PrintWriter writer = new PrintWriter(new FileWriter(RESULTS_FILE, true))) {
            writer.println(result.toString());
        } catch (IOException e) {
            System.err.println("CRITICAL: Failed to write result to log file: " + e.getMessage());
        }
    }

    /**
     * Helper to get the names of activation functions for logging.
     */
    private String getFunctionNames(IDifferentiableFunction[] functions) {
        List<String> names = new ArrayList<>();
        for (IDifferentiableFunction func : functions) {
            names.add(func.getClass().getSimpleName());
        }
        return String.join(", ", names);
    }

    /**
     * Parses a result object from a log line.
     */
    private TuningResult parseResultFromString(String line) {
        try {
            String params = line.substring(line.indexOf('[') + 1, line.indexOf(']'));
            String accString = line.substring(line.indexOf("Accuracy: ") + 10, line.indexOf('%'));
            double accuracy = Double.parseDouble(accString);
            return new TuningResult(params, accuracy);
        } catch (Exception e) { // Captura exceções mais genéricas (e.g., NumberFormatException)
            // Return a dummy result if parsing fails
            return new TuningResult(line, 0.0);
        }
    }

    /**
     * Entry point to run the optimizer.
     */
    public static void main(String[] args) {
        HyperparameterTuner tuner = new HyperparameterTuner();
        tuner.runGridSearch();
    }
}
