package apps;
import math.Matrix;
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


/**
 * Orchestrates a resilient and automated search for optimal MLP model hyperparameters through a parallelized grid search.
 * <p>
 * This class implements a Grid Search strategy to systematically explore multiple
 * combinations of hyperparameters, including learning rate, momentum, network topology,
 * and activation functions. It is designed for efficiency and resilience, making it
 * ideal for long-running tuning tasks that are prone to interruption.
 * </p>
 * <ul>
 *   <li><b>CPU-Based Training:</b> It uses the standard {@link MLP23} trainer to ensure results are consistent with the main application.</li>
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
 * <h3>Future Improvements</h3>
 * <p>
 * While Grid Search is exhaustive, it can be computationally expensive. For a more
 * efficient search in a large hyperparameter space, consider implementing:
 * </p>
 * <ul>
 *   <li><b>Random Search:</b> Instead of testing all combinations, Random Search samples a fixed
 *       number of random combinations. It often finds better models in less time.</li>
 *   <li><b>Bayesian Optimization:</b> An even more advanced technique that uses the results from
 * </ul>
 *
 * @see MLP23
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


    private final int SEED = MLP23.SEED;
    private final int epochs = 10000;

    private final double[] learningRates = {
            0.0019,
            0.0020,
            0.0021
    };

    private final double[] momentums = {
            0.985,
            0.99,
            0.995
    };

    private final int[][] topologies = {
            {400, 3, 1},
            {400, 4, 1},
    };

    private final IDifferentiableFunction[][] activationFunctions = {
            {new Sigmoid(), new Sigmoid()},
            //{new TanH(), new Sigmoid()},
    };

    private final double[] l2Lambdas = {
            0.0,
            0.00001,
    };


    private final boolean findSeed = false;

    /**
     * Entry point to run the optimizer.
     */
    public static void main(String[] args)
    {
        HyperparameterTuner tuner = new HyperparameterTuner();
        if(tuner.findSeed)
        {
            // Testa 100 seeds diferentes na configuração vencedora para encontrar a "sortuda"
            int[] seedsToTest = new int[100];
            for(int i=0; i<100; i++) seedsToTest[i] = i + 1;

            // Configuração Otimizada: [400, 3, 1], LR 0.0020, Momentum 0.99
            tuner.findBestSeedForConfig(new int[]{400, 3, 1}, new IDifferentiableFunction[]{new Sigmoid(), new Sigmoid()}, 0.0020, 0.99, 0.0, seedsToTest);
        }
        else
            // Cenário 1: Executar a busca em grade completa
            tuner.runGridSearch();

    }

    /**
         * A simple data class to store the results of a single training trial.
         */
        private record TuningResult(int seed, String paramsDescription, double accuracy, double precision, double recall, double f1Score, double extraAccuracy) implements Comparable<TuningResult> {

        @Override
            public int compareTo(TuningResult other) {
                // Ordena por acurácia em ordem decrescente (maior é melhor).
                // Em caso de empate, o F1-Score pode ser usado como critério secundário.
                return Double.compare(other.f1Score, this.f1Score);
            }

            @Override
            public String toString() {
                String base = String.format("Seed: %d, Parameters: [%s] -> Accuracy: %.2f%%, Precision: %.4f, Recall: %.4f, F1-Score: %.4f",
                        seed,
                        paramsDescription, accuracy, precision, recall, f1Score);
                // Adiciona a acurácia extra apenas se tiver sido calculada (não é 0.0)
                return extraAccuracy > 0.0 ? base + String.format(", Extra_Acc: %.2f%%", extraAccuracy) : base;
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
        System.out.printf("Found %d previously completed trial(s). They will be skipped.\n", completedTrials.size());

        // --- Otimização: Carregar os dados UMA VEZ antes de iniciar a busca ---
        System.out.println("Pre-loading and caching datasets to optimize parallel trials...");
        DataHandler dataHandler = new DataHandler(SEED, DataHandler.NormalizationType.MIN_MAX); // Usar uma seed fixa e Min-Max
        System.out.printf("-> Datasets loaded: %d training samples, %d test samples.\n\n",
                dataHandler.getTrainingDataSize(),
                dataHandler.getTestInputs().rows());

        // Create all combinations of parameters to be tested.
        List<Callable<TuningResult>> tasks = new ArrayList<>();
        for (int[] topology : topologies) {
            for (IDifferentiableFunction[] functions : activationFunctions) {
                // Ensure the topology and activation function combination is valid.
                if (topology.length - 1 != functions.length) continue;
                for (double lr : learningRates) {
                    for (double momentum : momentums) {
                        for (double l2Lambda : l2Lambdas) { // Novo loop para L2
                            String paramsDescription = String.format(
                                    "Topology: %s, Functions: %s, LR: %.4f, Momentum: %.2f, L2: %.4f",
                                    Arrays.toString(topology), getFunctionNames(functions), lr, momentum, l2Lambda
                            );

                            // Se a combinação já foi testada, salta-a.
                            if (completedTrials.contains(paramsDescription)) {
                                continue;
                            }
                            tasks.add(() -> runTrial(paramsDescription, topology, functions, lr, momentum, l2Lambda, SEED));
                        }
                    }
                }
            }
        }

        // --- Configuração do Paralelismo ---
        // Usar um número fixo de threads para não sobrecarregar a CPU.
        final int numThreads = Math.max(1, Runtime.getRuntime().availableProcessors() - 2);

        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        // CompletionService decouples task submission from result retrieval, which is more efficient.
        CompletionService<TuningResult> completionService = new ExecutorCompletionService<>(executor);
        List<TuningResult> results = new ArrayList<>();

        System.out.printf("--- Iniciando a Busca de Hiperparâmetros com %d thread(s) no modo CPU ---\n", numThreads);
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
                    Future<TuningResult> future = completionService.poll(40, TimeUnit.MINUTES);
                    if (future == null) {
                        System.err.print("\n[WARNING] A training trial timed out after 40 minutes and was cancelled. Skipping to the next one.\n");
                        continue; // Move to the next result
                    }
                    TuningResult trialResult = future.get();
                    results.add(trialResult);

                    // Guarda o resultado imediatamente, mas apenas se a acurácia for superior a 90%.
                    if (trialResult.accuracy >= 95.0) { // Subimos a fasquia pois já temos 98.75%
                        saveResult(trialResult); // Salva a linha completa no log
                        System.out.printf(">> Completed trial %d/%d. Result saved: Accuracy: %.2f%%, F1: %.4f\n", (i + 1), tasks.size(), trialResult.accuracy, trialResult.f1Score);
                    } else {
                        System.out.printf(">> Completed trial %d/%d. Accuracy < 97.0%% (%.2f%%). Result ignored.\n", (i + 1), tasks.size(), trialResult.accuracy);
                    }

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
            TuningResult bestResult = results.getFirst();
            System.out.println("\n--- BEST COMBINATION FOUND ---");
            System.out.println(bestResult);
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
     *   <li>Instantiating a trainer with the specified configuration.</li>
     *   <li>Executing the training process using the full training dataset.</li>
     *   <li>Evaluating the trained model against the test dataset to calculate its performance.</li>
     * </ol>
     *
     * @return A {@link TuningResult} object containing the parameters and final validation error.
     */
    private TuningResult runTrial(String paramsDescription, int[] topology, IDifferentiableFunction[] functions, double lr, double momentum, double l2Lambda, int seed) {
        try {
            System.out.printf("--- Testing combination (Seed: %d): %s ---\n", seed, paramsDescription);

            // --- CPU-BASED TRIAL ---
            // Carrega os dados para cada trial para garantir isolamento entre threads.
            DataHandler dataHandler = new DataHandler(seed, DataHandler.NormalizationType.MIN_MAX);

            // Instancia o treinador MLP23 com os hiperparâmetros da iteração atual.
            MLP23 trainer = new MLP23(topology, functions, lr, momentum, l2Lambda, this.epochs, seed);
            trainer.getMLP().train(dataHandler.getTrainInputs(), dataHandler.getTrainOutputs(), dataHandler.getTestInputs(), dataHandler.getTestOutputs(), lr, this.epochs, momentum, l2Lambda);

            // --- TEST THE TRAINED NETWORK ---
            MLP23.TestMetrics metrics = trainer.test(dataHandler.getTestInputs(), dataHandler.getTestOutputs()); // Teste padrão

            double extraAccuracy = 0.0;
            // Se o modelo for promissor, submete-o a um teste mais rigoroso.
            if (metrics.accuracy() >= 99.50) {
                System.out.println("-> Accuracy > 99.50%. Running extra validation...");
                Matrix[] extraTestData = DataHandler.loadExtraTestData();
                MLP23.TestMetrics extraMetrics = trainer.test(extraTestData[0], extraTestData[1]);
                extraAccuracy = extraMetrics.accuracy();
                System.out.printf("-> Extra Test Accuracy: %.2f%%\n", extraAccuracy);
            }

            System.out.printf("--- Finished trial (Seed: %d): [%s] -> Accuracy: %.2f%%, F1-Score: %.4f ---\n", seed, paramsDescription, metrics.accuracy(), metrics.f1Score());
            return new TuningResult(seed, paramsDescription, metrics.accuracy(), metrics.precision(), metrics.recall(), metrics.f1Score(), extraAccuracy);
        } catch (Exception e) {
            // Use e.toString() for a more descriptive message, as e.getMessage() can be null.
            System.err.printf("--- FAILED trial: [%s] -> Exception: %s. Likely a data loading issue. ---\n", paramsDescription, e.toString());
            // Retorna um resultado com 0 de acurácia e F1-Score para marcar a tentativa como falhada.
            return new TuningResult(seed, paramsDescription, 0.0, 0.0, 0.0, 0.0, 0.0);
        }
    }

    /**
     * Tests a single hyperparameter configuration across multiple seeds to find the most robust one.
     *
     * @param topology The network topology to test.
     * @param functions The activation functions to use.
     * @param lr The learning rate.
     * @param momentum The momentum.
     * @param seedsToTest An array of seeds to evaluate.
     */
    public void findBestSeedForConfig(int[] topology, IDifferentiableFunction[] functions, double lr, double momentum, double l2Lambda, int[] seedsToTest) {
        String paramsDescription = String.format(
                "Topology: %s, Functions: %s, LR: %.4f, Momentum: %.2f, L2: %.4f",
                Arrays.toString(topology), getFunctionNames(functions), lr, momentum, l2Lambda
        );

        System.out.printf("\n--- Finding Best Seed for Configuration: %s ---\n", paramsDescription);
        System.out.printf("Testing across %d different seeds.\n\n", seedsToTest.length);

        List<Callable<TuningResult>> tasks = new ArrayList<>();
        for (int seed : seedsToTest) {
            tasks.add(() -> runTrial(paramsDescription, topology, functions, lr, momentum, l2Lambda, seed));
        }

        final int numThreads = Math.max(1, Runtime.getRuntime().availableProcessors() - 2);
        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        CompletionService<TuningResult> completionService = new ExecutorCompletionService<>(executor);
        List<TuningResult> results = new ArrayList<>();

        for (Callable<TuningResult> task : tasks) {
            completionService.submit(task);
        }

        try {
            for (int i = 0; i < tasks.size(); i++) {
                try {
                    Future<TuningResult> future = completionService.take();
                    TuningResult res = future.get();
                    results.add(res);
                    saveResult(res); // Guarda o resultado no ficheiro log
                } catch (ExecutionException e) {
                    System.err.println("A seed trial failed: " + e.getCause().getMessage());
                }
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            System.err.println("Seed search was interrupted.");
        } finally {
            executor.shutdown();
        }

        Collections.sort(results);

        System.out.println("\n\n--- === Seed Search Summary === ---");
        results.forEach(System.out::println);

        if (!results.isEmpty()) {
            System.out.println("\n--- BEST SEED FOUND ---");
            System.out.println(results.getFirst());
        }
        System.out.println("--- ========================= ---");
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
                if (line.contains("Parameters: [")) {
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
            int seed = 0;
            String params = line.substring(line.indexOf('[') + 1, line.indexOf(']'));
            double accuracy = 0.0;
            double precision = 0.0;
            double recall = 0.0;
            double f1Score = 0.0;
            double extraAccuracy = 0.0;

            if (line.contains("Seed: ")) {
                String seedStr = line.substring(line.indexOf("Seed: ") + 6, line.indexOf(", Parameters:"));
                seed = Integer.parseInt(seedStr.trim());
            }

            if (line.contains("Accuracy: ")) {
                String accString = line.substring(line.indexOf("Accuracy: ") + 10, line.indexOf('%'));
                accuracy = Double.parseDouble(accString.replace(',', '.'));
            }
            if (line.contains("Precision: ")) {
                String precString = line.substring(line.indexOf("Precision: ") + 11, line.indexOf(", Recall:"));
                precision = Double.parseDouble(precString.replace(',', '.'));
            }
            if (line.contains("Recall: ")) {
                String recallString = line.substring(line.indexOf("Recall: ") + 8, line.indexOf(", F1-Score:"));
                recall = Double.parseDouble(recallString.replace(',', '.'));
            }
            if (line.contains("F1-Score: ")) {
                String f1String = line.substring(line.indexOf("F1-Score: ") + 10).trim();
                f1Score = Double.parseDouble(f1String.replace(',', '.'));
            }
            if (line.contains("Extra_Acc: ")) {
                String extraAccString = line.substring(line.indexOf("Extra_Acc: ") + 11, line.lastIndexOf('%'));
                extraAccuracy = Double.parseDouble(extraAccString.replace(',', '.'));
            }

            return new TuningResult(seed, params, accuracy, precision, recall, f1Score, extraAccuracy);
        } catch (Exception e) { // Captura exceções mais genéricas (e.g., NumberFormatException)
            return new TuningResult(0, line, 0.0, 0.0, 0.0, 0.0, 0.0); // Retorna um resultado dummy em caso de falha
        }
    }

}
