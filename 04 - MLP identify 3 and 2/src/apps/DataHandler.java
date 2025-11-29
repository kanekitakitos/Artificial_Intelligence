package apps;

import math.Matrix;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Collection;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.Random;

/**
 * A comprehensive utility class for managing and preprocessing datasets for machine learning models.
 * <p>
 * This class serves a dual purpose. It can be instantiated to manage the entire lifecycle of a dataset,
 * including loading, shuffling, splitting into training and validation sets, and preprocessing. It also
 * provides static methods for common, one-off data handling tasks.
 *
 * <h3>1. Instance-Based Dataset Management</h3>
 * <p>
 * When instantiated, {@code DataHandler} takes multiple input and output file paths and orchestrates them into
 * ready-to-use training and validation matrices. The process includes:
 * </p>
 * <ul>
 *   <li><b>Loading:</b> Reads data from multiple CSV files.</li>
 *   <li><b>Shuffling:</b> Randomizes the order of data points to prevent model bias.</li>
 *   <li><b>Splitting:</b> Divides the data into a training set and a smaller validation set based on a specified fraction.</li>
 *   <li><b>Preprocessing:</b> Applies normalization (e.g., scaling pixel values) and label encoding.</li>
 * </ul>
 *
 * <h4>Example: Loading and Splitting Data for Training</h4>
 * <pre>{@code
 * public class TrainModel {
 *     public static void main(String[] args) {
 *         String[] trainInputs = {"src/data/treino_inputs.csv"};
 *         String[] trainOutputs = {"src/data/treino_labels.csv"};
 *         double validationFraction = 0.2; // Use 20% of the data for validation
 *         int seed = 42;
 *
 *         // 1. Create a DataHandler instance to manage the dataset.
 *         DataHandler dataManager = new DataHandler(trainInputs, trainOutputs, validationFraction, seed);
 *
 *         // 2. Retrieve the matrices for training and validation.
 *         Matrix trainData = dataManager.getTrainInputs();
 *         Matrix trainLabels = dataManager.getTrainOutputs();
 *         Matrix valData = dataManager.getValidationInputs();
 *         Matrix valLabels = dataManager.getValidationOutputs();
 *
 *         // 3. (Proceed to train the model with these matrices...)
 *         // MLP mlp = new MLP(...);
 *         // mlp.train(trainData, trainLabels, ...);
 *     }
 * }
 * }</pre>
 *
 * <h3>2. Static Utility Methods</h3>
 * <p>
 * The class also offers static methods for direct data manipulation without needing an instance. These are useful for
 * simple, standalone tasks like loading a dedicated test set.
 * </p>
 *
 * <h4>Example: Loading a Test Dataset</h4>
 * <pre>{@code
 * // Directly load input and output files into matrices without splitting.
 * Matrix[] testData = DataHandler.loadTestData("src/data/test_inputs.csv", "src/data/test_labels.csv");
 * Matrix testInputs = testData[0];
 * Matrix testOutputs = testData[1];
 * }</pre>
 *
 * @see neural.MLP
 * @see Matrix
 * @author Brandon Mejia, hdaniel@ualg.pt
 * @version 2025-11-29
 */
public class DataHandler {

    private static class DataPoint {
        final double[] input;
        final double[] output;
        DataPoint(double[] input, double[] output) { this.input = input; this.output = output; }
    }


    private final Matrix trainInputs;
    private final Matrix trainOutputs;
    private final Matrix validationInputs;
    private final Matrix validationOutputs;
    private int trainingDataSize;
    private int validationDataSize;

    /**
     * Constructor for loading separate training and validation/test sets.
     * @deprecated Use the constructor with a split ratio for more flexibility.
     */
    @Deprecated
    public DataHandler(String[] trainInputPaths, String[] trainOutputPaths, String valInputPath, String valOutputPath, int seed) {
        List<DataPoint> trainingData = loadAndProcess(trainInputPaths, trainOutputPaths);
        Collections.shuffle(trainingData, new Random(seed));
        this.trainInputs = new Matrix(listTo2DArray(trainingData, true));
        this.trainOutputs = new Matrix(listTo2DArray(trainingData, false));
        this.trainingDataSize = trainingData.size();

        List<DataPoint> validationData = loadAndProcess(new String[]{valInputPath}, new String[]{valOutputPath});
        this.validationInputs = new Matrix(listTo2DArray(validationData, true));
        this.validationOutputs = new Matrix(listTo2DArray(validationData, false));
        this.validationDataSize = validationData.size();
    }

    /**
     * Creates a DataHandler by loading all data, shuffling it, and then splitting it
     * into a training set and a validation/test set based on the provided ratio.
     *
     * @param allInputPaths   An array of paths to all input data files (CSV).
     * @param allOutputPaths  An array of paths to all output label files (CSV).
     * @param validationSplit The percentage of the data to be used for validation (e.g., 0.2 for 20%).
     * @param seed            The seed for the random number generator to ensure reproducibility.
     */
    public DataHandler(String[] allInputPaths, String[] allOutputPaths, double validationSplit, int seed) {
        // 1. Load all data points from all files
        List<DataPoint> allData = loadAndProcess(allInputPaths, allOutputPaths);

        // 2. Shuffle the entire dataset randomly
        Collections.shuffle(allData, new Random(seed));

        // 3. Split the data into training and validation sets
        int validationSize = (int) (allData.size() * validationSplit);
        this.validationDataSize = validationSize;
        this.trainingDataSize = allData.size() - validationSize;

        List<DataPoint> validationData = allData.subList(0, validationSize);
        List<DataPoint> trainingData = allData.subList(validationSize, allData.size());

        // 4. Convert to matrices
        this.trainInputs = new Matrix(listTo2DArray(trainingData, true));
        this.trainOutputs = new Matrix(listTo2DArray(trainingData, false));
        this.validationInputs = new Matrix(listTo2DArray(validationData, true));
        this.validationOutputs = new Matrix(listTo2DArray(validationData, false));
    }

    private List<DataPoint> loadAndProcess(String[] inputPaths, String[] outputPaths) {
        // Processa cada par de ficheiros em paralelo e recolhe as listas de DataPoints.
        return IntStream.range(0, inputPaths.length).parallel()
                .mapToObj(i -> {
                    String inputPath = inputPaths[i];
                    // Usa o último ficheiro de labels se não houver um correspondente.
                    String outputPath = (i < outputPaths.length) ? outputPaths[i] : outputPaths[outputPaths.length - 1];

                    List<double[]> inputs = loadCsv(inputPath);
                    List<double[]> outputs = loadCsv(outputPath);

                    if (inputs.size() != outputs.size()) {
                        throw new IllegalStateException("Input file " + inputPath + " and output file " + outputPath + " have a different number of rows.");
                    }

                    // Processa as linhas de cada ficheiro e cria os DataPoints.
                    return IntStream.range(0, inputs.size())
                            .mapToObj(j -> {
                                double[] input = inputs.get(j);
                                // Normaliza o input para o intervalo [0, 1] de forma mais eficiente.
                                for (int k = 0; k < input.length; k++) input[k] /= 255.0;
                                double[] output = outputs.get(j);
                                output[0] = (output[0] == 3.0) ? 1.0 : 0.0; // Converte 3.0 para 1.0, e o resto para 0.0
                                return new DataPoint(input, output);
                            }).collect(Collectors.toList());
                })
                .flatMap(Collection::stream) // Agrega todas as listas de DataPoints numa só.
                .collect(Collectors.toList());
    }

    private static List<double[]> loadCsv(String filePath) {
        List<double[]> data = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] stringValues = line.split(",");
                double[] doubleValues = new double[stringValues.length];
                for (int i = 0; i < stringValues.length; i++) {
                    doubleValues[i] = Double.parseDouble(stringValues[i]);
                }
                data.add(doubleValues);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return data;
    }

    private static double[][] listTo2DArray(List<DataPoint> dataPoints, boolean isInput) {
        if (dataPoints.isEmpty()) return new double[0][0];
        int numRows = dataPoints.size();
        int numCols = isInput ? dataPoints.get(0).input.length : dataPoints.get(0).output.length;
        double[][] array = new double[numRows][numCols];
        for (int i = 0; i < numRows; i++) {
            array[i] = isInput ? dataPoints.get(i).input : dataPoints.get(i).output;
        }
        return array;
    }

    /**
     * Carrega um conjunto de dados de teste a partir dos caminhos de ficheiro fornecidos.
     * Este método não embaralha os dados, preservando a sua ordem original para avaliação.
     *
     * @param inputPath  O caminho para o ficheiro de inputs (features) do teste.
     * @param outputPath O caminho para o ficheiro de outputs (labels) do teste.
     * @return Um array de {@link Matrix} contendo as matrizes de input e output. O índice 0 contém os inputs e o índice 1 os outputs.
     */
    public static Matrix[] loadTestData(String inputPath, String outputPath)
    {
        // 1. Carrega os dados brutos dos ficheiros CSV.
        List<double[]> inputs = loadCsv(inputPath);
        List<double[]> outputs = loadCsv(outputPath);

        if (inputs.size() != outputs.size()) {
            throw new IllegalStateException("Ficheiro de input " + inputPath + " e ficheiro de output " + outputPath + " têm um número de linhas diferente.");
        }

        // 2. Pré-processa os dados e cria os DataPoints.
        List<DataPoint> testData = IntStream.range(0, inputs.size()).parallel()
                .mapToObj(i -> {
                    double[] input = inputs.get(i);
                    for (int k = 0; k < input.length; k++) input[k] /= 255.0; // Normaliza
                    double[] output = outputs.get(i);
                    output[0] = (output[0] == 3.0) ? 1.0 : 0.0; // Converte labels
                    return new DataPoint(input, output);
                }).collect(Collectors.toList());

        // 3. Converte para matrizes e retorna.
        Matrix testInputs = new Matrix(listTo2DArray(testData, true));
        Matrix testOutputs = new Matrix(listTo2DArray(testData, false));
        return new Matrix[]{testInputs, testOutputs};
    }

    public Matrix getTrainInputs() { return trainInputs; }
    public Matrix getTrainOutputs() { return trainOutputs; }
    public Matrix getValidationInputs() { return validationInputs; }
    public Matrix getValidationOutputs() { return validationOutputs; }
    public int getTrainingDataSize() { return trainingDataSize; }
    public int getValidationDataSize() { return validationDataSize; }
}