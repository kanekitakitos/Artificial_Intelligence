package test;

import math.Matrix;
import apps.MLP23;
import neural.MLP;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

@DisplayName("MLP Test for identifying digits 2 and 3")
class TestMLP23 {

    private static MLP mlp;
    private static Matrix testX;
    private static Matrix testY;


    /**
     * Loads data from the CSV file.
     * The first 100 rows are images of the digit '2' (label 0).
     * The next rows are images of the digit '3' (label 1).
     */
    private static void loadData()
    {
        String testFile = "src/data/test.csv";
        List<double[]> featureList = new ArrayList<>(); // Changed from MLP to mlp
        List<double[]> labelList = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(testFile))) {
            String line;
            int lineCount = 0;
            while ((line = br.readLine()) != null) {
                String[] stringValues = line.split(",");
                double[] features = new double[stringValues.length];
                for (int i = 0; i < stringValues.length; i++) {
                    features[i] = Double.parseDouble(stringValues[i]);
                }
                featureList.add(features);

                // The first 100 rows are '2' (label 0), the rest are '3' (label 1)
                labelList.add(new double[]{(lineCount < 100) ? 0 : 1});
                lineCount++;
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        testX = new Matrix(featureList.toArray(new double[0][]));
        testY = new Matrix(labelList.toArray(new double[0][]));
    }

    @BeforeAll
    static void setUp() {
        // Load the test data
        loadData();

        // Get a pre-trained clone of the MLP model
        MLP23 modelFactory = new MLP23();

        // Caminhos para os seus arquivos de dados
        String[] inputPaths = {
                //"src/data/dataset.csv",
                "src/data/datasetExtreme.csv",
                //"src/data/dataset_apenas_novos.csv",
                //"src/data/dataset_apenas_novos2.csv",
        };
        String[] outputPaths = {"src/data/labels.csv"};//,"src/data/labels.csv","src/data/labels.csv","src/data/labels.csv"};
        modelFactory.train(inputPaths, outputPaths);
        mlp = modelFactory.getMLP();
    }

    @Test
    @DisplayName("Should achieve high accuracy on the test set")
    void testModelAccuracy() {
        // Cria um arquivo para registrar os resultados detalhados do teste
        try (PrintWriter writer = new PrintWriter(new FileWriter("src/data/test_results.txt"))) {
            int correctPredictions = 0;
            writer.println("--- Análise de Predição do Teste ---");
            writer.println("------------------------------------");

            for (int i = 0; i < testY.rows(); i++) {
                Matrix inputRow = getRowAsMatrix(testX, i);
                Matrix prediction = mlp.predict(inputRow);
                double rawPrediction = prediction.get(0, 0);
                long predictedLabel = Math.round(rawPrediction);
                double actualValue = testY.get(i, 0);
                boolean isCorrect = predictedLabel == actualValue;

                if (isCorrect) {
                    correctPredictions++;
                }

                // Escreve o resultado de cada teste no arquivo
                writer.printf("Índice: %-4d | Esperado: %.0f | Previsto: %-25.17f | Resultado: %s\n",
                        i, actualValue, rawPrediction, isCorrect ? "CORRETO" : "INCORRETO");
            }

            double accuracy = (double) correctPredictions / testY.rows();
            System.out.printf("Model Accuracy: %.2f%%\n", accuracy * 100);
            writer.println("------------------------------------");
            writer.printf("\nAcurácia Final: %.2f%%\n", accuracy * 100);

            // Assert que a acurácia seja maior que um certo limiar, por exemplo, 95%
            assertTrue(accuracy > 0.96, "A acurácia do modelo deve ser maior que 95%");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Test
    @DisplayName("Should correctly predict a '2'")
    void testSinglePredictionForDigitTwo() {
        // Using the first image, which is a '2' (label 0)
        Matrix inputForTwo = getRowAsMatrix(testX, 0);
        Matrix prediction = mlp.predict(inputForTwo);
        double rawPrediction = prediction.get(0, 0);
        long predictedLabel = Math.round(rawPrediction);

        System.out.printf("Predição para '2' (esperado ~0.0): %.8f\n", rawPrediction);
        assertEquals(0, predictedLabel, "The MLP should predict '0' for an image of a '2'");
    }

    @Test
    @DisplayName("Should correctly predict a '3'")
    void testSinglePredictionForDigitThree() {
        // Using the 101st image, which is a '3' (label 1)
        Matrix inputForThree = getRowAsMatrix(testX, 100);
        Matrix prediction = mlp.predict(inputForThree);
        double rawPrediction = prediction.get(0, 0);
        long predictedLabel = Math.round(rawPrediction);

        System.out.printf("Predição para '3' (esperado ~1.0): %.8f\n", rawPrediction);
        assertEquals(1, predictedLabel, "The MLP should predict '1' for an image of a '3'");
    }

    /**
     * Helper method to extract a single row from a matrix and return it as a new 1xN Matrix.
     * @param originalMatrix The matrix to extract the row from.
     * @param rowIndex The index of the row to extract.
     * @return A new 1xN matrix containing the data from the specified row.
     */
    private Matrix getRowAsMatrix(Matrix originalMatrix, int rowIndex) {
        double[] row = new double[originalMatrix.cols()];
        for (int j = 0; j < originalMatrix.cols(); j++) {
            row[j] = originalMatrix.get(rowIndex, j);
        }
        return new Matrix(new double[][]{row});
    }
}