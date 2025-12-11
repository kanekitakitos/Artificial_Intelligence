package apps;

import math.Matrix;
import neural.MLP;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * A comprehensive data management utility for loading, preprocessing, and serving datasets for the MLP.
 * <p>
 * This class centralizes all dataset handling logic, providing a single point of access for training,
 * validation, and test data. It is designed to be instantiated with specific configurations
 * (like the validation split ratio) and then serve ready-to-use matrices to the models.
 * The file paths for the datasets are hardcoded as constants, promoting cleaner code in
 * the training and hyperparameter tuning classes.
 * </p>
 *
 * <h3>Core Functionality</h3>
 * <ul>
 *   <li><b>Centralized Path Management:</b> Default paths for training and test data are managed internally.</li>
 *   <li><b>Automated Preprocessing:</b> Includes robust pixel value normalization (from [0, 255] to [0, 1])
 *       and binary label encoding (e.g., digit '3' becomes 1.0, others become 0.0).</li>
 *   <li><b>Data Aggregation:</b> Loads data from multiple source files into a single, unified dataset.</li>
 *   <li><b>Data Shuffling & Splitting:</b> Automatically shuffles the dataset with a fixed seed for reproducibility
 *       and can split it into training and validation sets.</li>
 * </ul>
 *
 * <h3>Example: Training a Model with Default Data</h3>
 * <p>
 * To train a model, simply create a {@code DataHandler} instance and retrieve the necessary matrices.
 * This decouples the model training logic from the specifics of data file locations.
 * </p>
 * <pre>{@code
 * // 1. Instantiate DataHandler with a 20% validation split.
 * DataHandler dataManager = new DataHandler(0.2, 42);
 *
 * // 2. Get the processed data matrices.
 * Matrix trainInputs = dataManager.getTrainInputs();
 * Matrix trainOutputs = dataManager.getTrainOutputs();
 * Matrix validationInputs = dataManager.getValidationInputs();
 * Matrix validationOutputs = dataManager.getValidationOutputs();
 *
 * // 3. Train the model.
 * MLP mlp = new MLP(new int[]{400, 2, 1}, new IDifferentiableFunction[]{new Sigmoid(), new Sigmoid()});
 * mlp.train(trainInputs, trainOutputs, 0.01, 100, 0.8);
 *
 * // 4. Load test data for final evaluation.
 * Matrix[] testSet = DataHandler.loadDefaultTestData();
 * double accuracy = calculateAccuracy(mlp, testSet[0], testSet[1]);
 * }
 * }</pre>
 *
 * @see MLP
 * @see Matrix
 * @author Brandon Mejia, hdaniel@ualg.pt
 * @version 2025-11-30
 */
public class DataHandler {

    // --- Default File Paths ---
    private static final String[] DEFAULT_INPUT_PATHS = {
            "src/data/moonshak/datasetBig.csv",
            //"src/data/train/test_apenas_novos2.csv",

    };
    private static final String[] DEFAULT_OUTPUT_PATHS = {
            "src/data/moonshak/labelsBig.csv", // Reused for all input files
            //"src/data/train/testL.csv",
    };

    //private static final String DEFAULT_TEST_INPUT_PATH = "src/data/minist/test_pixels.csv";
    //private static final String DEFAULT_TEST_LABELS_PATH = "src/data/minist/test_labels.csv";

    private static final String DEFAULT_TEST_INPUT_PATH = "src/data/testData/datasetTest.csv";
    private static final String DEFAULT_TEST_LABELS_PATH = "src/data/testData/labelsTest.csv";

    // --- Caminhos para o conjunto de dados de teste extra (para validação rigorosa) ---
    private static final String EXTRA_TEST_INPUT_PATH = "src/data/minist/test_pixels.csv";
    private static final String EXTRA_TEST_LABELS_PATH = "src/data/minist/test_labels.csv";

    public enum NormalizationType {
        MIN_MAX, // Normaliza para o intervalo [0, 1]
        Z_SCORE  // Normaliza usando a média e o desvio padrão
    }

    private static class DataPoint {
        final double[] input;
        final double[] output;
        DataPoint(double[] input, double[] output) { this.input = input; this.output = output; }
    }


    private Matrix trainInputs;
    private Matrix trainOutputs;
    private Matrix validationInputs;
    private Matrix validationOutputs;
    private int trainingDataSize;
    private int validationDataSize;

    /**
     * Construtor principal que carrega, processa e divide os dados.
     *
     * @param seed O seed para baralhar os dados de forma reprodutível.
     * @param normType O tipo de normalização a ser aplicado aos dados de entrada (MIN_MAX ou Z_SCORE).
     */
    public DataHandler(int seed, NormalizationType normType) {
        // Carrega e processa os dados de treino
        List<DataPoint> trainingData = loadAndProcess(DEFAULT_INPUT_PATHS, DEFAULT_OUTPUT_PATHS, normType, null, null);
        Collections.shuffle(trainingData, new Random(seed));
        this.trainInputs = new Matrix(listTo2DArray(trainingData, true));
        this.trainOutputs = new Matrix(listTo2DArray(trainingData, false));
        this.trainingDataSize = trainingData.size();

        // Carrega e processa os dados de validação/teste
        List<DataPoint> validationData = loadAndProcess(new String[]{DEFAULT_TEST_INPUT_PATH}, new String[]{DEFAULT_TEST_LABELS_PATH}, normType, null, null);
        this.validationInputs = new Matrix(listTo2DArray(validationData, true));
        this.validationOutputs = new Matrix(listTo2DArray(validationData, false));
        this.validationDataSize = validationData.size();
    }



    /**
     * Constructor for loading separate training and validation/test sets.
     * @deprecated Use the constructor with a split ratio for more flexibility.
     */
    @Deprecated
    public DataHandler(String[] trainInputPaths, String[] trainOutputPaths, String valInputPath, String valOutputPath, int seed) {
        List<DataPoint> trainingData = loadAndProcess(trainInputPaths, trainOutputPaths, NormalizationType.MIN_MAX, null, null);
        Collections.shuffle(trainingData, new Random(seed));
        this.trainInputs = new Matrix(listTo2DArray(trainingData, true));
        this.trainOutputs = new Matrix(listTo2DArray(trainingData, false));
        this.trainingDataSize = trainingData.size();

        List<DataPoint> validationData = loadAndProcess(new String[]{valInputPath}, new String[]{valOutputPath}, NormalizationType.MIN_MAX, null, null);
        this.validationInputs = new Matrix(listTo2DArray(validationData, true));
        this.validationOutputs = new Matrix(listTo2DArray(validationData, false));
        this.validationDataSize = validationData.size();
    }

    /**
     * Creates a DataHandler by loading all data, shuffling it, and then splitting it
     * into a training set and a validation set based on the provided ratio.
     * <p>
     * This constructor orchestrates the entire data preparation pipeline:
     * <ol>
     *   <li>Loads data from all specified file paths.</li>
     *   <li><b>Shuffles the entire dataset</b> to ensure that training and validation sets are representative samples.</li>
     *   <li>Splits the shuffled data into training and validation subsets according to the {@code validationSplit} ratio.</li>
     *   <li>Converts these subsets into the final {@link Matrix} objects ready for model training.</li>
     * </ol>
     *
     * @param allInputPaths   An array of paths to all input data files (CSV).
     * @param allOutputPaths  An array of paths to all output label files (CSV).
     * @param validationSplit The percentage of the data to be used for validation (e.g., 0.2 for 20%).
     */
    public DataHandler(String[] allInputPaths, String[] allOutputPaths, double validationSplit, int seed) {
        // 1. Load all data points from all files
        List<DataPoint> allData = loadAndProcess(allInputPaths, allOutputPaths, NormalizationType.MIN_MAX, null, null);

        // 2. Shuffle the entire dataset randomly
        //Collections.shuffle(allData, new Random(seed));
        Collections.shuffle(allData, new Random(seed));
        
        if (validationSplit > 0.0) {
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
        } else {
            // If validationSplit is 0, use all data for training.
            this.trainInputs = new Matrix(listTo2DArray(allData, true));
            this.trainOutputs = new Matrix(listTo2DArray(allData, false));
            this.validationInputs = new Matrix(new double[0][0]); // Empty matrix
            this.validationOutputs = new Matrix(new double[0][0]); // Empty matrix
        }
    }

    /**
     * Loads and preprocesses data from multiple file pairs in parallel.
     * <p>
     * This method is responsible for the core ETL (Extract, Transform, Load) logic:
     * <ul>
     *   <li><b>Extract:</b> Reads raw data from CSV files using parallel streams for efficiency.</li>
     *   <li><b>Transform:</b>
     *     <ul>
     *       <li>Normalizes input features by scaling them from the [0, 255] range to [0, 1]. It intelligently
     *           skips this step if the data already appears to be normalized (i.e., all values are <= 1.0).</li>
     *       <li>Encodes output labels, mapping the target class (3.0) to 1.0 and other classes to 0.0.</li>
     *     </ul>
     *   </li>
     *   <li><b>Load:</b> Aggregates the processed records into a unified list of {@code DataPoint} objects.</li>
     * </ul>
     */
    private List<DataPoint> loadAndProcess(String[] inputPaths, String[] outputPaths, NormalizationType normType, double[] precomputedMean, double[] precomputedStdDev) {
        // --- 1. Carregar todos os dados brutos ---
        List<double[]> allInputs = new ArrayList<>();
        List<double[]> allOutputs = new ArrayList<>();

        for (int i = 0; i < inputPaths.length; i++) {
            String inputPath = inputPaths[i];
            String outputPath = (i < outputPaths.length) ? outputPaths[i] : outputPaths[outputPaths.length - 1];
            List<double[]> inputs = loadCsv(inputPath);
            List<double[]> outputs = loadCsv(outputPath);

            if (inputs.size() != outputs.size()) {
                throw new IllegalStateException("Mismatch in rows between " + inputPath + " and " + outputPath);
            }
            allInputs.addAll(inputs);
            allOutputs.addAll(outputs);
        }

        // --- 2. Aplicar Normalização ---
        if (normType == NormalizationType.Z_SCORE) {
            int numFeatures = allInputs.get(0).length;
            double[] mean = (precomputedMean != null) ? precomputedMean : new double[numFeatures];
            double[] stdDev = (precomputedStdDev != null) ? precomputedStdDev : new double[numFeatures];

            // Calcular média e desvio padrão se não forem fornecidos
            if (precomputedMean == null || precomputedStdDev == null) {
                // Calcular média
                for (double[] input : allInputs) {
                    for (int j = 0; j < numFeatures; j++) {
                        mean[j] += input[j];
                    }
                }
                for (int j = 0; j < numFeatures; j++) {
                    mean[j] /= allInputs.size();
                }

                // Calcular desvio padrão
                for (double[] input : allInputs) {
                    for (int j = 0; j < numFeatures; j++) {
                        stdDev[j] += Math.pow(input[j] - mean[j], 2);
                    }
                }
                for (int j = 0; j < numFeatures; j++) {
                    stdDev[j] = Math.sqrt(stdDev[j] / allInputs.size());
                    if (stdDev[j] == 0) stdDev[j] = 1; // Evitar divisão por zero
                }
            }

            // Aplicar normalização Z-score
            for (double[] input : allInputs) {
                for (int j = 0; j < numFeatures; j++) {
                    input[j] = (input[j] - mean[j]) / stdDev[j];
                }
            }
        } else { // MIN_MAX (comportamento original)
            for (double[] input : allInputs) {
                boolean needsNormalization = Arrays.stream(input).anyMatch(val -> val > 1.0);
                if (needsNormalization) {
                    for (int k = 0; k < input.length; k++) {
                        input[k] /= 255.0;
                    }
                }
            }
        }

        // --- 3. Processar outputs e criar DataPoints ---
        List<DataPoint> dataPoints = new ArrayList<>();
        for (int i = 0; i < allInputs.size(); i++) {
            double[] input = allInputs.get(i);
            double[] output = allOutputs.get(i);
            // Converte as etiquetas para o formato binário (0 ou 1) para distinguir entre 2 e 3.
            // Se a etiqueta já for 0 ou 1, assume-se que já está no formato correto.
            if (output[0] == 3.0) {
                output[0] = 1.0; // O dígito '3' é a classe positiva.
            } else if (output[0] == 2.0) {
                output[0] = 0.0; // O dígito '2' é a classe negativa.
            }
            dataPoints.add(new DataPoint(input, output));
        }

        return dataPoints;
    }

    private static List<double[]> loadCsv(String filePath)
    {
        List<double[]> data = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(filePath)))
        {
            String line;
            while ((line = br.readLine()) != null)
            {
                String[] stringValues = line.split(",");
                double[] doubleValues = new double[stringValues.length];
                for (int i = 0; i < stringValues.length; i++)
                    doubleValues[i] = Double.parseDouble(stringValues[i]);

                data.add(doubleValues);
            }
        }
        catch (IOException e)
        {
            e.printStackTrace();
        }
        return data;
    }

    private static double[][] listTo2DArray(List<DataPoint> dataPoints, boolean isInput)
    {
        if (dataPoints.isEmpty()) return new double[0][0];
        int numRows = dataPoints.size();
        int numCols = isInput ? dataPoints.get(0).input.length : dataPoints.get(0).output.length;
        double[][] array = new double[numRows][numCols];
        for (int i = 0; i < numRows; i++)
            array[i] = isInput ? dataPoints.get(i).input : dataPoints.get(i).output;

        return array;
    }

    /**
     * Carrega um conjunto de dados de teste a partir dos caminhos de ficheiro fornecidos.
     * Este método não embaralha os dados, preservando a sua ordem original para avaliação.
     *
     * @param inputPath  O caminho para o ficheiro de inputs (features) do teste.
     * @param labelsPath O caminho para o ficheiro de outputs (labels) do teste.
     * @return Um array de {@link Matrix} contendo as matrizes de input e output. O índice 0 contém os inputs e o índice 1 os outputs.
     */
    public static Matrix[] loadTestData(String inputPath, String labelsPath)
    {
        // 1. Carrega os dados brutos dos ficheiros CSV.
        List<double[]> inputs = loadCsv(inputPath);
        List<double[]> outputs = loadCsv(labelsPath);

        if (inputs.size() != outputs.size())
            throw new IllegalStateException("Ficheiro de input " + inputPath);


        // 2. Pré-processa os dados e cria os DataPoints.
        List<DataPoint> testData = IntStream.range(0, inputs.size()).parallel()
                .mapToObj(i -> {
                    double[] input = inputs.get(i);
                    // Production-Ready Check: Only normalize if data appears to be in the [0, 255] pixel range.
                    boolean needsNormalization = Arrays.stream(input).anyMatch(val -> val > 1.0);
                    if (needsNormalization) {
                        for (int k = 0; k < input.length; k++) {
                            input[k] /= 255.0;
                        }
                    }

                    double[] output = outputs.get(i);
                    // Converte as etiquetas para o formato binário (0 ou 1) para distinguir entre 2 e 3.
                    // Se a etiqueta já for 0 ou 1, assume-se que já está no formato correto.
                    if (output[0] == 3.0) {
                        output[0] = 1.0; // O dígito '3' é a classe positiva.
                    } else if (output[0] == 2.0) {
                        output[0] = 0.0; // O dígito '2' é a classe negativa.
                    }
                    return new DataPoint(input, output);
                }).collect(Collectors.toList());

        // 3. Converte para matrizes e retorna.
        Matrix testInputs = new Matrix(listTo2DArray(testData, true));
        Matrix testOutputs = new Matrix(listTo2DArray(testData, false));
        return new Matrix[]{testInputs, testOutputs};
    }

    /**
     * Carrega o conjunto de dados de teste predefinido.
     *
     * @return Um array de {@link Matrix} onde o índice 0 é a matriz de inputs e o índice 1 é a de outputs.
     */
    public static Matrix[] loadDefaultTestData() {
        return loadTestData(DEFAULT_TEST_INPUT_PATH, DEFAULT_TEST_LABELS_PATH);
    }

    /**
     * Carrega o conjunto de dados de teste "extra" para uma validação mais rigorosa.
     *
     * @return Um array de {@link Matrix} onde o índice 0 são os inputs e o índice 1 os outputs.
     */
    public static Matrix[] loadExtraTestData() {
        return loadTestData(EXTRA_TEST_INPUT_PATH, EXTRA_TEST_LABELS_PATH);
    }

    public Matrix getTrainInputs() { return trainInputs; }
    public Matrix getTrainOutputs() { return trainOutputs; }
    public Matrix getTestInputs() { return validationInputs; }
    public Matrix getTestOutputs() { return validationOutputs; }
    public int getTrainingDataSize() { return trainingDataSize; }
    public int getValidationDataSize() { return validationDataSize; }
}