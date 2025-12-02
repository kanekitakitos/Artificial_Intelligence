package apps;

import math.Matrix;
import neural.activation.*;
import neural.activation.IDifferentiableFunction;
import neural.MLP;

/**
 * Encapsulates the entire configuration and training process for a specific Multi-Layer Perceptron (MLP) model.
 * <p>
 * This class acts as a high-level trainer for an {@link MLP}. It defines the network's architecture (topology and activation functions),
 * sets the hyperparameters (learning rate, epochs, momentum), and manages the training lifecycle. The training process
 * is now encapsulated within the {@link MLP#train(Matrix, Matrix, Matrix, Matrix, double, int, double)} method,
 * which includes features like asynchronous validation and best-model checkpointing.
 * It relies on the {@link DataHandler} to load, preprocess, and split the datasets for training and validation.
 *
 * <h3>Example Usage</h3>
 * <p>
 * The following example demonstrates how to instantiate this class, train the model, and then use the resulting
 * best-performing MLP to make predictions on a new, unseen test set.
 * </p>
 *
 * <h4>Training and Evaluating the Model</h4>
 * <pre>{@code
 * public class Main {
 *     public static void main(String[] args) {
 *         // 1. Define the paths for the training data.
 *         String[] trainInputs = {"src/data/treino_inputs.csv"};
 *         String[] trainOutputs = {"src/data/treino_labels.csv"};
 *
 *         // 2. Create an instance of the trainer and execute the training process.
 *         MLP23 trainer = new MLP23();
 *         trainer.train(trainInputs, trainOutputs);
 *
 *         // 3. Retrieve the best-performing MLP after training is complete.
 *         MLP bestModel = trainer.getBestMLP();
 *
 *         // 4. Load a separate, unseen test dataset to evaluate the model.
 *         Matrix[] testData = DataHandler.loadTestData("src/data/test.csv", "src/data/labelsTest.csv");
 *         Matrix testInputs = testData[0];
 *         Matrix testOutputs = testData[1];
 *
 *         // 5. Make predictions on the test data.
 *         Matrix predictions = bestModel.predict(testInputs);
 *
 *         // 6. Print the first 5 predictions vs actual values.
 *         System.out.println("--- Test Results (Prediction vs Actual) ---");
 *         for (int i = 0; i < 5; i++) {
 *             double predictedValue = predictions.get(i, 0) > 0.5 ? 1.0 : 0.0; // Convert probability to binary class
 *             double actualValue = testOutputs.get(i, 0);
 *             System.out.printf("Sample %d: Predicted=%.1f, Actual=%.1f\n", i, predictedValue, actualValue);
 *         }
 *     }
 * }
 * }</pre>
 *
 * @see MLP
 * @see DataHandler
 * @see IDifferentiableFunction
 * @author Brandon Mejia
 * @version 2025-11-29
 */
public class MLP23 {

    private double lr = 0.01;

    // bigRuido and borroso
    private int epochs = 20000;
    private double momentum = 0.80;
    private int[] topology = {400,1, 1};
    private IDifferentiableFunction[] functions = {new Sigmoid(), new Sigmoid()};
    private MLP mlp;
    private static final int SEED = 8; // 2;4;5 5:00 ;7;8 4:21 ;16 4:17


    /**
     * Constructs the MLP trainer with a predefined network topology and activation functions.
     */
    public MLP23()
    {
        this.mlp = new MLP(topology, functions, SEED);
    }

    public MLP23(int[] topology, IDifferentiableFunction[] functions, double lr, double momentum, int epochs) {
        this.topology = topology;
        this.functions = functions;
        this.lr = lr;
        this.momentum = momentum;
        this.epochs = epochs;
        this.mlp = new MLP(this.topology, this.functions, SEED);
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
        DataHandler dataManager = new DataHandler(SEED);
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
        return this.mlp.train(trainInputs, trainOutputs, valInputs, valOutputs, this.lr, this.epochs, this.momentum);
    }

    /**
     * Retrieves the fully trained Multi-Layer Perceptron model.
     * <p>This is the best-performing model found during the training process, selected based on the lowest validation error.</p>
     * @return The trained {@link MLP} instance.
     */
    public MLP getMLP() { return this.mlp; }


}