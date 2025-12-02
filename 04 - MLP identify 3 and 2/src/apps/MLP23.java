package apps;

import math.Matrix;
import neural.activation.*;
import neural.activation.IDifferentiableFunction;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import neural.MLP;

/**
 * Encapsulates the entire configuration and training process for a specific Multi-Layer Perceptron (MLP) model.
 * <p>
 * This class acts as a high-level trainer for an {@link MLP}. It defines the network's architecture (topology and activation functions),
 * sets the hyperparameters (learning rate, epochs, momentum), and manages the training lifecycle. The training process
 * is enhanced with advanced techniques such as:
 * <ul>
 *     <li><b>Asynchronous Validation:</b> Performs validation on a separate thread to avoid blocking the training loop.</li>
 *     <li><b>Best Model Checkpointing:</b> Automatically saves the model with the lowest validation error found so far.</li>
 *     <li><b>Adaptive Learning Rate:</b> Reduces the learning rate if the validation error stops improving.</li>
 *     <li><b>Early Stopping:</b> Halts the training process if the validation error fails to improve for a specified number of epochs, preventing overfitting.</li>
 * </ul>
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
    private int epochs = 100000;
    private int internalEpochs = 1;
    private double momentum = 0.60;
    private int[] topology = {400,2, 1};
    private IDifferentiableFunction[] functions = {new Sigmoid(), new Sigmoid()};
    private MLP mlp;
    private static final int SEED = 8; // 2;4;5 5:00 ;7;8 4:21 ;16 4:17


    /**
     * Constructs the MLP trainer with a predefined network topology and activation functions.
     */
    public MLP23() {
        // A inicialização aleatória de pesos é feita dentro do construtor da MLP.
        // A inicialização a zero foi removida para permitir que a rede aprenda.
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
    public void train() {
        // 1. Load and prepare the data using DataHandler
        DataHandler dataManager = new DataHandler(SEED);
        // 2. Call the core training method with the prepared matrices
        train(dataManager.getTrainInputs(), dataManager.getTrainOutputs(), dataManager.getValidationInputs(), dataManager.getValidationOutputs());
    }

    /**
     * A high-level training method that loads separate datasets for training and validation.
     * <p>
     * This overload is ideal when you have pre-separated files for training and validation.
     * It uses the {@link DataHandler} to load and preprocess each dataset independently,
     * without performing any data splitting.
     * </p>
     *
     * @param trainInputPaths  An array of file paths for the input training data.
     * @param trainOutputPaths An array of file paths for the corresponding training label data.
     * @param valInputPath     The file path for the validation input data.
     * @param valOutputPath    The file path for the validation label data.
     */
    public void train(String[] trainInputPaths, String[] trainOutputPaths, String valInputPath, String valOutputPath) {
        // 1. Load and prepare the data using the DataHandler constructor for separate sets.
        // Note: This constructor does not split the data; it loads train and validation sets from distinct files.
        DataHandler dataManager = new DataHandler(trainInputPaths, trainOutputPaths, valInputPath, valOutputPath, SEED);
        // 2. Call the core training method with the prepared matrices.
        train(dataManager.getTrainInputs(), dataManager.getTrainOutputs(), dataManager.getValidationInputs(), dataManager.getValidationOutputs());
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
    public double train(Matrix trainInputs, Matrix trainOutputs, Matrix valInputs, Matrix valOutputs) {
        System.out.println("Iniciando o treinamento da rede...");
        System.out.println("Amostras de Treino: " + trainInputs.rows() + " | Amostras de Validação: " + valInputs.rows());

        // Executor para tarefas assíncronas de validação
        ExecutorService validationExecutor = Executors.newSingleThreadExecutor();
        CompletableFuture<Double> validationFuture = null;


        // 2. Lógica de treino
        double bestValidationError = Double.POSITIVE_INFINITY;
        MLP bestMlp = null; // Para guardar o melhor modelo
        int epochsSinceLastErrorIncrease = 0;
        final int lrPatience = 10; // Paciência para reduzir a learning rate
        final int earlyStoppingPatience = 50; // Paciência para parar o treino

        for (int epoch = 1; epoch <= this.epochs; epoch++) {
            this.mlp.train(trainInputs, trainOutputs, this.lr, internalEpochs,momentum);

            // A cada 10 épocas, calcula o erro de validação
            if (epoch % 10 == 0)
            {
                // Espera que a validação anterior termine, se existir
                if (validationFuture != null)
                {
                    try {
                        double currentValidationError = validationFuture.get(); // Obtém o resultado do cálculo anterior

                        // Imprime o erro de validação (MSE) a cada 100 épocas para acompanhar o progresso.
                        if ((epoch - 10) > 0 && (epoch - 10) % 100 == 0)
                        {
            //                //System.out.printf("Época: %-5d | LR: %.6f | Erro de Validação (MSE): %.6f\n", epoch - 10, this.lr, currentValidationError);
                        }

                        if (currentValidationError < bestValidationError)
                        {
                            bestValidationError = currentValidationError;
                            epochsSinceLastErrorIncrease = 0;

                            // Guarda uma cópia do melhor modelo encontrado
                            bestMlp = this.mlp.clone();
                        }
                        else
                            epochsSinceLastErrorIncrease++;


                        // Se o erro de validação não melhora, reduz o LR
                        if (epochsSinceLastErrorIncrease > 0 && epochsSinceLastErrorIncrease % (lrPatience / 10) == 0)
                        {
                            // Redução mais suave da learning rate para evitar saltos
                            // diminui o lr um 10%
                            this.lr *= 0.90;
                            epochsSinceLastErrorIncrease = 0; // Reset do contador
                        }
                    }
                    catch (Exception e)
                    {
                        e.printStackTrace();
                    }
                }

                // Lança a próxima validação de forma assíncrona
                final MLP modelCloneForValidation = this.mlp.clone();
                validationFuture = CompletableFuture.supplyAsync(() ->
                {
                    Matrix valPrediction = modelCloneForValidation.predict(valInputs);
                    return valOutputs.sub(valPrediction).apply(x -> x * x).sum() / valInputs.rows();
                }, validationExecutor);

                // 2. Condição de paragem se a learning rate ficar muito pequena
                if (this.lr < 1e-5)
                {
                    //System.out.println("Learning rate muito baixa. A parar o treino.");
                    break;
                }
                // 3. Condição de Early Stopping
                if (epochsSinceLastErrorIncrease >= earlyStoppingPatience / 10)
                {
                    //System.out.printf("\n--- Early Stopping ativado na época %d ---\n", epoch);
                    //System.out.println("O erro de validação não melhora há muito tempo.");
                    break;
                }
            }
        }

        validationExecutor.shutdown(); // Desliga o executor
        System.out.println("Treinamento concluído.");

        // 4. Restaurar o melhor modelo que foi guardado
        if (bestMlp != null)
            this.mlp = bestMlp;
            //System.out.printf("\nMelhor modelo restaurado (com erro de validação de: %.6f)\n", bestValidationError);

        return bestValidationError;
    }

    /**
     * Retrieves the fully trained Multi-Layer Perceptron model.
     * <p>This is the best-performing model found during the training process, selected based on the lowest validation error.</p>
     * @return The trained {@link MLP} instance.
     */
    public MLP getMLP() { return this.mlp; }


}