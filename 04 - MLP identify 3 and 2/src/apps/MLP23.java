package apps;

import java.io.IOException;
import math.Matrix;
import neural.activation.IDifferentiableFunction;
import neural.activation.Sigmoid;
import neural.MLP;

/**
 * A factory and trainer for a specific Multi-Layer Perceptron (MLP) model designed to classify handwritten digits '2' and '3'.
 * <p>
 * This class encapsulates a pre-defined MLP configuration, including its topology, activation functions, and hyperparameters
 * like learning rate and momentum. It acts as a high-level trainer that simplifies the process of preparing and training
 * the model by orchestrating the {@link DataHandler} for data loading and the {@link MLP} for the core training execution.
 * </p>
 * <p>
 * The primary training logic, which includes asynchronous validation and best-model checkpointing, is handled by the
 * {@link MLP#train(Matrix, Matrix, Matrix, Matrix, double, int, double)} method. This class serves as a convenient
 * wrapper to instantiate and run that process with a proven configuration.
 * </p>
 *
 * <h3>Example Usage</h3>
 * <p>
 * This class is designed to be used programmatically as a factory for a trained model. The main application entry
 * point is now in the {@code P4} class. The following example demonstrates how to use {@code MLP23} to get a
 * trained model and use it for predictions.
 * </p>
 *
 * <pre>{@code
 * // 1. Instantiate the trainer factory.
 * MLP23 trainer = new MLP23();
 *
 * // 2. Execute the training process. This loads and processes data automatically.
 * trainer.train();
 *
 * // 3. Retrieve the best-performing model after training.
 * MLP bestModel = trainer.getMLP();
 *
 * // 4. (Optional) Save the trained model for later use.
 * String modelPath = "src/data/models/my_trained_model.ser";
 * try {
 *     bestModel.saveModel(modelPath);
 * } catch (IOException e) {
 *     e.printStackTrace();
 * }
 *
 * // 5. Use the model to make a prediction on a new data sample.
 * // (Assuming 'newImageMatrix' is a 1x400 Matrix).
 * Matrix prediction = bestModel.predict(newImageMatrix);
 * long label = Math.round(prediction.get(0, 0));
 * System.out.println(label == 0 ? "Predicted: 2" : "Predicted: 3");
 * }</pre>
 *
 * @see MLP#saveModel(String)
 * @see MLP#loadModel(String)
 * @see MLP
 * @see DataHandler
 * @see IDifferentiableFunction
 * @author Brandon Mejia
 * @version 2025-12-02
 */
public class MLP23
{

    private final double lr = 0.01;
    private final int epochs = 20000;
    private final double momentum = 0.9;
    private MLP mlp;
    public static final int SEED =1; // 2;4;5 5:00 ;7;8 4:21 ;16 4:17
    // seed 8 bigRuido e dataset 99.25%

    /**
     * Constructs the MLP trainer with a predefined network topology and activation functions.
     */
    public MLP23()
    {
        IDifferentiableFunction[] functions = {new Sigmoid(), new Sigmoid()};
        int[] topology = {400,1, 1};

        this.mlp = new MLP(topology, functions, SEED);
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


    /**
     * Trains the model and saves it to a default file path.
     * This is a convenience method that calls {@link #trainAndSaveModel(String)}.
     */
    public void saveMLP()
    {
        // Define o caminho padrão para guardar o modelo.
        String modelPath = "src/data/models/digit_classifier_v"+99+"_dataset_seed1.ser";
        trainAndSaveModel(modelPath);
    }

    /**
     * Trains the model and then saves the final state to the specified file path.
     * @param modelPath The path where the trained model will be saved (e.g., "src/data/models/my_model.ser").
     */
    public void trainAndSaveModel(String modelPath) {
        System.out.println("--- Iniciando processo de treino e gravação ---");
        train(); // Executa o treino completo.
        try {
            this.mlp.saveModel(modelPath);
        } catch (IOException e) {
            System.err.println("ERRO: Falha ao guardar o modelo em " + modelPath);
            e.printStackTrace();
        }
    }
}