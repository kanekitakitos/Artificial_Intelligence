import apps.DataHandler;
import apps.MLP23;
import math.Matrix;
import neural.MLP;
import neural.ModelUtils;
import neural.activation.IDifferentiableFunction;
import neural.activation.Sigmoid;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Arrays;

/**
 * The main entry point for the MLP-based digit classification program, designed for automated evaluation.
 * <p>
 * This class orchestrates the process by first training a dedicated {@link MLP23} model.
 * After training is complete, it enters a loop to read multiple lines of input from the standard input stream.
 * Each line is expected to contain 400 comma-separated pixel values representing a 20x20 image.
 * </p>
 * <p>
 * For each input image, it predicts whether the digit is a '2' or a '3' and prints the result to the
 * standard output, followed by a new line. This process continues until the input stream is closed.
 * </p>
 *
 * @author Brandon Mejia
 * @version 2025-12-03
 */
public class P4 {

    /**
     * Trains the model and processes multiple input samples for classification.
     *
     * @param args Command-line arguments (not used).
     * @throws IOException If an I/O error occurs while reading from the console.
     */
    public static void main(String[] args) throws IOException {
        // 1. Define o caminho para o modelo treinado.



        String modelPath = "src/models/digit_classifier_v99_dataset_seed1.ser";
        //trainMLP23(modelPath);
        MLP mlp = ModelUtils.loadModel(modelPath);

        // Se o modelo não pôde ser carregado, encerra a execução.
        if (mlp == null) {
            System.err.println("ERRO: O modelo MLP não foi carregado. Encerrando o programa.");
            return;
        }

        // 2. Read multiple lines from the console and predict each one.
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(System.in))) {
            String line;
            while ((line = reader.readLine()) != null && !line.isEmpty()) {
                double[] inputValues = Arrays.stream(line.split(","))
                        .map(String::trim)
                        .mapToDouble(Double::parseDouble)
                        .toArray();

                Matrix inputMatrix = new Matrix(new double[][]{inputValues});
                Matrix predictionMatrix = mlp.predict(inputMatrix);
                long predictedLabel = Math.round(predictionMatrix.get(0, 0));
                System.out.println(predictedLabel == 0 ? 2 : 3);
            }
        }
    }

    public static void trainMLP23(String modelPath)
    {
        double lr = 0.01;
        int epochs = 20000;
        double momentum = 0.9;
        int SEED = 1;
        IDifferentiableFunction[] functions = {new Sigmoid(), new Sigmoid()};
        int[] topology = {400,1, 1};
        MLP mlp = new MLP(topology, functions, SEED);
        DataHandler dataManager = new DataHandler(SEED);


        Matrix trainInputs = dataManager.getTrainInputs();
        Matrix trainOutputs = dataManager.getTrainOutputs();
        Matrix valInputs = dataManager.getTestInputs();
        Matrix valOutputs = dataManager.getTestOutputs();



        mlp.train(trainInputs, trainOutputs, valInputs, valOutputs, lr,epochs,momentum);
        mlp.saveModel(modelPath);
    }
}