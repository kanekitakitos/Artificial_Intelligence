

import apps.DataHandler;
import math.Matrix;
import apps.MLP23;
import neural.MLP;
import neural.ModelUtils;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Arrays;
import java.io.FileWriter;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Contains a suite of JUnit 5 tests to validate the performance and correctness of the {@link MLP23} model.
 * <p>
 * This class is responsible for ensuring that the trained neural network meets quality standards. It includes:
 * </p>
 * <ul>
 *   <li><b>Model Training:</b> A single, consistent model is trained once in the {@code @BeforeAll} setup method to ensure all tests run against the same baseline.</li>
 *   <li><b>Accuracy and Metrics Evaluation:</b> A comprehensive test ({@link #testModelAccuracy()}) evaluates the model on a standard test set, calculating key metrics like accuracy, precision, recall, and F1-score, and generating a detailed report.</li>
 *   <li><b>Large-Scale Validation:</b> An additional test ({@link #testModelWithExtraData()}) evaluates the model against a larger, independent dataset to verify its generalization capabilities.</li>
 *   <li><b>Unit Tests for Specific Predictions:</b> Isolated tests ({@link #testSinglePredictionForDigitTwo()} and {@link #testSinglePredictionForDigitThree()}) confirm that the model correctly classifies specific, known samples.</li>
 * </ul>
 * <p>
 * The results of the main evaluation are saved to {@code src/data/test_results.txt}, providing a traceable record of model performance.
 * </p>
 *
 * @see MLP23
 * @see DataHandler
 * @author Brandon Mejia
 * @version 2025-12-07
 */
@DisplayName("MLP Test for identifying digits 2 and 3")
class TestMLP23
{

    private static MLP mlp;
    private static Matrix testX;
    private static Matrix testY;
    private static final int SEED = 8;
    private static final String MODEL_PATH = P4.path;
    private static DataHandler dh = new DataHandler(SEED, DataHandler.NormalizationType.MIN_MAX);

    @BeforeAll
    static void setUp()
    {
        // Load the test data
        // Use DataHandler to load and preprocess the test data consistently.
        testX = dh.getTestInputs();
        testY = dh.getTestOutputs();

        // Para garantir que os testes sejam independentes e reproduzíveis,
        // treinamos um novo modelo a partir do zero antes de cada execução de teste.
        // Isso evita a dependência de um ficheiro de modelo pré-existente.



        MLP23 mlp23 = new MLP23();
        mlp23.train();
        //ModelUtils.saveModel(mlp23.getMLP(), MODEL_PATH+".model");
        ModelUtils.saveModelToJson(mlp23.getMLP(), MODEL_PATH);


        mlp = ModelUtils.loadModelFromJson(MODEL_PATH+".json");

    }

    @Test
    @DisplayName("Should achieve high accuracy on the test set")
    void testModelAccuracy()
    {
        // Cria um arquivo para registrar os resultados detalhados do teste
        try (PrintWriter writer = new PrintWriter(new FileWriter("src/data/test_results.txt")))
        {
            int correctPredictions = 0;
            int truePositives = 0;
            int falsePositives = 0;
            int falseNegatives = 0;
            int trueNegatives = 0;
            writer.println("--- Análise de Predição do Teste ---");
            writer.println("------------------------------------");

            for (int i = 0; i < testY.rows(); i++)
            {
                Matrix inputRow = getRowAsMatrix(testX, i);
                Matrix prediction = mlp.predict(inputRow);
                double rawPrediction = prediction.get(0, 0);
                long predictedLabel = Math.round(rawPrediction);
                double actualValue = testY.get(i, 0);
                boolean isCorrect = predictedLabel == actualValue;

                if (isCorrect) {
                    correctPredictions++;
                }

                // Calcula TP, FP, FN, TN (considerando '1' como a classe positiva)
                if (predictedLabel == 1 && actualValue == 1) {
                    truePositives++;
                } else if (predictedLabel == 1 && actualValue == 0) {
                    falsePositives++;
                } else if (predictedLabel == 0 && actualValue == 1) {
                    falseNegatives++;
                } else if (predictedLabel == 0 && actualValue == 0) {
                    trueNegatives++;
                }

                // Escreve o resultado de cada teste no arquivo
                writer.printf("Índice: %-4d | Esperado: %.0f | Previsto: %-25.17f | Resultado: %s\n",
                        i, actualValue, rawPrediction, isCorrect ? "CORRETO" : "INCORRETO");
            }

            double accuracy = (double) correctPredictions / testY.rows();

            // Calcula Precisão, Recall e F1-Score
            double precision = (truePositives + falsePositives > 0) ? (double) truePositives / (truePositives + falsePositives) : 0.0;
            double recall = (truePositives + falseNegatives > 0) ? (double) truePositives / (truePositives + falseNegatives) : 0.0;
            double f1Score = (precision + recall > 0) ? 2 * (precision * recall) / (precision + recall) : 0.0;

            // Imprime as métricas na consola
            System.out.println("\n--- Métricas de Avaliação do Modelo ---");
            System.out.printf("Acurácia: %.2f%%\n", accuracy * 100);
            System.out.printf("Precisão: %.4f\n", precision);
            System.out.printf("Recall (Sensibilidade): %.4f\n", recall);
            System.out.printf("F1-Score: %.4f\n", f1Score);
            System.out.println("---------------------------------------");

            // Escreve as métricas no final do arquivo
            writer.println("------------------------------------");
            writer.println("\n--- Resumo das Métricas ---");
            writer.printf("Acurácia Final: %.2f%%\n", accuracy * 100);
            writer.printf("Precisão: %.4f\n", precision);
            writer.printf("Recall: %.4f\n", recall);
            writer.printf("F1-Score: %.4f\n", f1Score);
            writer.printf("\nVerdadeiros Positivos (TP): %d\nFalsos Positivos (FP): %d\nVerdadeiros Negativos (TN): %d\nFalsos Negativos (FN): %d\n",
                    truePositives, falsePositives, trueNegatives, falseNegatives);

            // Assert que a acurácia seja maior que um certo limiar, por exemplo, 96.5%
            assertTrue(accuracy >= 0.965, "A acurácia do modelo deve ser maior que 95%");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Test
    @DisplayName("Should maintain high accuracy on the extra large test set")
    void testModelWithExtraData() {
        System.out.println("\n--- Running evaluation on EXTRA test data ---");
        Matrix[] extraTestData = DataHandler.loadExtraTestData();
        Matrix extraTestX = extraTestData[0];
        Matrix extraTestY = extraTestData[1];

        // Define os caminhos para os novos arquivos que guardarão as falhas
        String failedInputsPath = "src/data/failed.csv";
        String failedLabelsPath = "src/data/failedL.csv";

        try (PrintWriter failedInputsWriter = new PrintWriter(new FileWriter(failedInputsPath), true);
             PrintWriter failedLabelsWriter = new PrintWriter(new FileWriter(failedLabelsPath), true);
             PrintWriter summaryWriter = new PrintWriter(new FileWriter("src/data/extra_test_results.txt"))) {

            // Instancia um novo treinador e carrega o estado do modelo treinado
            MLP23 tester = new MLP23();
            tester.getMLP().setWeights(mlp.getWeights());
            tester.getMLP().setBiases(mlp.getBiases());

            // Chama o novo método de teste, passando os writers para registrar as falhas
            MLP23.TestMetrics metrics = tester.test(extraTestX, extraTestY, failedInputsWriter, failedLabelsWriter);

            // Imprime as métricas na consola
            System.out.printf("Acurácia no conjunto extra: %.2f%%\n", metrics.accuracy());
            System.out.printf("Precisão no conjunto extra: %.4f\n", metrics.precision());
            System.out.printf("Recall no conjunto extra: %.4f\n", metrics.recall());
            System.out.printf("F1-Score no conjunto extra: %.4f\n", metrics.f1Score());
            System.out.println("Testes que falharam foram guardados em: " + failedInputsPath + " e " + failedLabelsPath);
            System.out.println("---------------------------------------------");

            // Escreve o resumo no arquivo de resultados
            summaryWriter.println("--- Resumo das Métricas (Conjunto Extra) ---");
            summaryWriter.printf("Acurácia Final: %.2f%%\n", metrics.accuracy());
            summaryWriter.printf("Precisão: %.4f\n", metrics.precision());
            summaryWriter.printf("Recall: %.4f\n", metrics.recall());
            summaryWriter.printf("F1-Score: %.4f\n", metrics.f1Score());
            
            assertTrue(metrics.accuracy() >= 95.0, "A acurácia do modelo no conjunto de dados extra deve ser maior que 95%");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Test
    @DisplayName("Diagnose model with Log Loss and Noise Robustness")
    void diagnosticarModelo() {
        Matrix inputs = testX;
        Matrix targets = testY;

        double logLoss = 0.0;
        int acertosRuido = 0;
        double somaPesos = 0.0;

        // Calcular a soma dos pesos absolutos do modelo
        for (Matrix w_layer : mlp.getWeights()) {
            somaPesos += w_layer.abs().sum();
        }

        for (int i = 0; i < inputs.rows(); i++) {
            double[] input = inputs.get(i); // Extrai a linha diretamente
            Matrix inputRowMatrix = new Matrix(new double[][]{input});
            double target = targets.get(i, 0);

            // 1. Previsão Normal para Log Loss
            Matrix predictionMatrix = mlp.predict(inputRowMatrix);
            double output = predictionMatrix.get(0, 0);
            // Proteção contra log(0) ou log(1)
            output = Math.max(1e-15, Math.min(1 - 1e-15, output));

            logLoss += (target == 1) ? -Math.log(output) : -Math.log(1 - output);

            // 2. Teste com Ruído
            double[] inputRuido = new double[input.length];
            for (int j = 0; j < input.length; j++) {
                // Adiciona +/- 5% de ruído absoluto
                inputRuido[j] = input[j] + ((Math.random() * 0.1) - 0.05);
            }
            Matrix inputRuidoMatrix = new Matrix(new double[][]{inputRuido});
            double outputRuido = mlp.predict(inputRuidoMatrix).get(0, 0);
            int previsaoRuido = (outputRuido >= 0.5) ? 1 : 0;
            if (previsaoRuido == (int) target) acertosRuido++;
        }

        System.out.println("\n--- DIAGNÓSTICO PROFUNDO DO MODELO ---");
        System.out.printf("1. Log Loss (Menor é Melhor): %.6f\n", (logLoss / inputs.rows()));
        System.out.printf("2. Acurácia com Ruído (5%%): %.2f%%\n", ((double) acertosRuido / inputs.rows() * 100));
        System.out.printf("3. Soma dos Pesos (Magnitude): %.2f\n", somaPesos);
        System.out.println("------------------------------------");
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