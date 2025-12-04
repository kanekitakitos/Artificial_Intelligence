import apps.DataHandler;
import apps.MLP23;
import math.Matrix;
import neural.MLP;
import neural.ModelUtils;
import neural.activation.IDifferentiableFunction;
import neural.activation.Sigmoid;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;

/**
 * The main entry point for the MLP-based digit classification program, designed for batch processing and automated evaluation.
 * <p>
 * This class orchestrates the process by first training a dedicated {@link MLP23} model.
 * After training is complete, it enters a loop to read multiple lines of input from the standard input stream.
 * Each line is expected to contain 400 comma-separated pixel values representing a 20x20 image.
 * </p>
 * <p>
 * For each input image, it predicts whether the digit is a '2' or a '3' and prints the result to the standard output,
 * followed by a new line. This process continues until the input stream is closed.
 * </p>
 *
 * @author Brandon Mejia
 * @version 2025-12-02
 */
public class P4 {
    /**
     * Loads a pre-trained model, reads all image data from standard input,
     * performs batch prediction, and prints the results.
     *
     * @param args Command-line arguments (not used).
     * @throws IOException If an I/O error occurs.
     */
    public static void main(String[] args) throws IOException
    {
        // 1. Create and train the model once.
        String path = "src/models/mlp23vSeed1Dataset1Neuron.model";
        //trainMLP23(path);
        MLP mlp = ModelUtils.loadModel(path);

        // 2. Read all 'm' samples from standard input.
        List<double[]> allInputs = new ArrayList<>();
        try (Scanner scanner = new Scanner(System.in)) {
            String line;
            while (scanner.hasNextLine() && !(line = scanner.nextLine()).trim().isEmpty())
            {
                double[] inputValues = Arrays.stream(line.split(","))
                        .map(String::trim)
                        .mapToDouble(Double::parseDouble)
                        .toArray();
                allInputs.add(inputValues);
            }
        }
        

        if (!allInputs.isEmpty())
        {
            // Convert the list of arrays to a 2D array (m x 400).
            int numRows = allInputs.size();
            int numCols = 400;
            double[][] inputArray = new double[numRows][numCols];
            for (int i = 0; i < numRows; i++)
            {
                // Ensure each row has exactly 400 columns, copying the values.
                System.arraycopy(allInputs.get(i), 0, inputArray[i], 0, Math.min(allInputs.get(i).length, numCols));
            }
            Matrix inputMatrix = new Matrix(inputArray);
            Matrix predictions = mlp.predict(inputMatrix);

            // 4. Print each result on a new line.
            for (int i = 0; i < predictions.rows(); i++)
            {
                long predictedLabel = Math.round(predictions.get(i, 0));
                System.out.println(predictedLabel == 0 ? 2 : 3);
            }
        }
    }

    /**
     * Trains a new MLP model with a specific topology and saves it to the specified path.
     *
     * @param modelPath The file path where the trained model will be saved.
     */
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