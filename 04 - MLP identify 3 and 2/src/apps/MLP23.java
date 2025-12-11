package apps;

import math.Matrix;
import neural.MLP;
import neural.activation.IDifferentiableFunction;

import java.io.PrintWriter;
import java.util.Arrays;
import java.util.stream.Collectors;
import neural.activation.Sigmoid;
import neural.activation.TanH;

/**
 * A factory and trainer for a Multi-Layer Perceptron (MLP) model designed to classify handwritten digits '2' and '3'.
 * <p>
 * This class simplifies the process of training and evaluating the model by orchestrating the
 * {@link DataHandler} for data loading and the {@link MLP} for the core training execution.
 * It serves as a convenient wrapper to instantiate and run the training process, making it
 * easy to integrate with hyperparameter search tools like {@link HyperparameterTuner}.
 * </p>
 *
 * <h3>Example Usage</h3>
 * <p>
 * This class can be used to train a model with a specific configuration and then evaluate it.
 * </p>
 * <pre>{@code
 * // 1. Define the model configuration.
 * int[] topology = {400, 4, 1};
 * IDifferentiableFunction[] functions = {new Sigmoid(), new Sigmoid()};
 * double learningRate = 0.0025;
 * double momentum = 0.99;
 * double l2Lambda = 0.0001;
 * int epochs = 20000;
 * int seed = 8;
 *
 * // 2. Instantiate the trainer.
 * MLP23 trainer = new MLP23(topology, functions, learningRate, momentum, l2Lambda, epochs, seed);
 *
 * // 3. Execute the training process.
 * trainer.train();
 *
 * // 4. Retrieve the trained model for further use or evaluation.
 * MLP bestModel = trainer.getMLP();
 *      }</pre>
 *
 * @see MLP
 * @see DataHandler
 * @see IDifferentiableFunction
 * @see HyperparameterTuner
 *
 * @author Brandon Mejia
 * @version 2025-12-02
 */
public class MLP23
{

    private final double lr;
    private final int epochs;
    private final double momentum;
    private final MLP mlp;
    private final int seed;
    private final double l2Lambda;
    public static final int SEED = 8; // 2;4;5 5:00 ;7;8 4:21 ;16 4:17
    // 17 seed

    /**
     * Constructs the MLP trainer with a predefined network topology and activation functions.
     * This default constructor uses a known good configuration for quick testing.
     */
    public MLP23()
    {

        this(new int[]{400, 2, 1}, new IDifferentiableFunction[]{new Sigmoid(), new Sigmoid()}, 0.0010, 0.99,0.00, 10000, SEED);
    }

    /**
     * Constructs the MLP trainer with a specific network configuration and hyperparameters.
     * This constructor is ideal for hyperparameter tuning.
     *
     * @param topology The network layer sizes.
     * @param functions The activation functions for each layer transition.
     * @param lr The learning rate.
     * @param momentum The momentum factor.
     * @param l2Lambda The L2 regularization factor (lambda).
     * @param epochs The number of training epochs.
     * @param seed The seed for random weight initialization.
     */
    public MLP23(int[] topology, IDifferentiableFunction[] functions, double lr, double momentum, double l2Lambda, int epochs, int seed)
    {
        if (topology.length - 1 != functions.length) {
            throw new IllegalArgumentException("The number of activation functions must be one less than the number of layers.");
        }
        this.lr = lr;
        this.momentum = momentum;
        this.l2Lambda = l2Lambda;
        this.epochs = epochs;
        this.seed = seed;
        this.mlp = new MLP(topology, functions, this.seed);
    }

    /**
     * @deprecated Use the constructor that explicitly requires a seed and l2Lambda.
     */
    @Deprecated
    public MLP23(int[] topology, IDifferentiableFunction[] functions, double lr, double momentum, int epochs) {
        this(topology, functions, lr, momentum, 0.0, epochs, SEED);
    }

    /**
     * A high-level training method that orchestrates the entire process from file paths.
     * <p>
     * This method serves as a convenient entry point for training. It uses the {@link DataHandler}
     * to load, preprocess, and split the data into training and validation sets before
     * initiating the core training loop.
     * </p>
     */
    public void train()
    {
        // 1. Load and prepare the data using DataHandler
        DataHandler dataManager = new DataHandler(this.seed, DataHandler.NormalizationType.MIN_MAX);
        // 2. Call the core training method with the prepared matrices
        train(dataManager.getTrainInputs(), dataManager.getTrainOutputs(), dataManager.getTestInputs(), dataManager.getTestOutputs());
    }

    /**
     * Trains the MLP model using the provided training and validation datasets.
     *
     * @param trainInputs  The matrix of training input data.
     * @param trainOutputs The matrix of training label data.
     * @param valInputs    The matrix of validation input data.
     * @param valOutputs   The matrix of validation label data.
     * @return The best validation error (MSE) achieved during training.
     */
    public double train(Matrix trainInputs, Matrix trainOutputs, Matrix valInputs, Matrix valOutputs)
    {
        // The complex training logic is now encapsulated within the MLP class itself.
        return this.mlp.train(trainInputs, trainOutputs, valInputs, valOutputs, this.lr, this.epochs, this.momentum, this.l2Lambda);
    }

    /**
     * A simple data class to hold test evaluation metrics.
     */
    public record TestMetrics(double accuracy, double precision, double recall, double f1Score) {}

    /**
     * Evaluates the trained model against a test dataset, computes performance metrics, and logs incorrect predictions.
     *
     * @param testInputs The test input data.
     * @param testOutputs The test target labels.
     * @param failedInputsWriter A writer for the input data of failed predictions. Can be null.
     * @param failedLabelsWriter A writer for the labels of failed predictions. Can be null.
     * @return A {@link TestMetrics} object containing accuracy, precision, recall, and F1-score.
     */
    public TestMetrics test(Matrix testInputs, Matrix testOutputs, PrintWriter failedInputsWriter, PrintWriter failedLabelsWriter) {
        int truePositives = 0;
        int falsePositives = 0;
        int falseNegatives = 0;
        int correctPredictions = 0;

        for (int i = 0; i < testOutputs.rows(); i++) {
            double[] row = new double[testInputs.cols()];
            for (int j = 0; j < testInputs.cols(); j++) {
                row[j] = testInputs.get(i, j);
            }
            Matrix inputRow = new Matrix(new double[][]{row});

            Matrix prediction = mlp.predict(inputRow);
            long predictedLabel = Math.round(prediction.get(0, 0));
            double actualValue = testOutputs.get(i, 0);

            if (predictedLabel == actualValue) {
                correctPredictions++;
            } else {
                // Log the failed prediction if writers are provided
                if (failedInputsWriter != null && failedLabelsWriter != null) {
                    // Converte o array de double para uma string CSV
                    String inputAsString = Arrays.stream(row)
                            .mapToObj(String::valueOf)
                            .collect(Collectors.joining(","));
                    failedInputsWriter.println(inputAsString);
                    failedLabelsWriter.println((int) actualValue);
                }
            }

            if (predictedLabel == 1 && actualValue == 1) truePositives++;
            else if (predictedLabel == 1 && actualValue == 0) falsePositives++;
            else if (predictedLabel == 0 && actualValue == 1) falseNegatives++;
        }

        double accuracy = (double) correctPredictions / testOutputs.rows() * 100.0;
        double precision = (truePositives + falsePositives > 0) ? (double) truePositives / (truePositives + falsePositives) : 0.0;
        double recall = (truePositives + falseNegatives > 0) ? (double) truePositives / (truePositives + falseNegatives) : 0.0;
        double f1Score = (precision + recall > 0) ? 2 * (precision * recall) / (precision + recall) : 0.0;

        return new TestMetrics(accuracy, precision, recall, f1Score);
    }

    /**
     * Evaluates the trained model against a test dataset and computes performance metrics.
     *
     * @param testInputs The test input data.
     * @param testOutputs The test target labels.
     * @return A {@link TestMetrics} object containing accuracy, precision, recall and F1-score.
     */
    public TestMetrics test(Matrix testInputs, Matrix testOutputs) {
        return test(testInputs, testOutputs, null, null);
    }

    /**
     * Retrieves the fully trained Multi-Layer Perceptron model.
     * <p>This is the best-performing model found during the training process, selected based on the lowest validation error.</p>
     * @return The trained {@link MLP} instance.
     */
    public MLP getMLP() { return this.mlp; }
}