import apps.MLP23;
import math.Matrix;
import neural.MLP;
import neural.ModelUtils;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;

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
 * @version 2025-12-02
 */
public class P4
{
    public static String path = "src/models/mlp23";

    /**
     * Loads a pre-trained model, reads all image data from standard input,
     * performs batch prediction, and prints the results.
     *
     * @param args Command-line arguments (not used).
     */
    public static void main(String[] args)
    {
        MLP mlp = ModelUtils.loadModelFromJson(path+".json");

        // 2. Read all 'm' samples from standard input.
        List<double[]> allInputs = new ArrayList<>();
        try (Scanner scanner = new Scanner(System.in))
        {
            String line;
            while (scanner.hasNextLine() && !(line = scanner.nextLine()).trim().isEmpty())
            {
                double[] inputValues = Arrays.stream(line.split(","))
                        .map(String::trim)
                        .mapToDouble(Double::parseDouble)
                        // IMPORTANT: Normalize the input data in the same way as the training data.
                        .map(pixel -> pixel / 255.0) // Example: Normalizing from [0, 255] to [0, 1]
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


    public static void trainMLP23(String modelPath)
    {
        MLP23 mlp23 = new MLP23();
        mlp23.train();
        MLP mlp = mlp23.getMLP();
        //ModelUtils.saveModel(mlp, modelPath+".model");

        ModelUtils.saveModelToJson(mlp, modelPath+".json");



    }
}