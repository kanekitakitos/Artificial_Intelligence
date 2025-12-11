package neural;

import math.Matrix;
import neural.activation.IDifferentiableFunction;
import neural.activation.Sigmoid;
import neural.activation.TanH;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Scanner;

/**
 * A utility class for handling model serialization and deserialization.
 * <p>
 * This class provides static methods to save and load {@link MLP} models to and from the file system.
 * It centralizes the file I/O logic for model persistence, ensuring a consistent approach across the application.
 * </p>
 * <h3>Example Usage</h3>
 * <pre>{@code
 * // To save a model:
 * MLP myModel = new MLP(...);
 * ModelUtils.saveModel(myModel, "src/models/my_model.ser");
 *
 * // To load a model:
 * MLP loadedModel = ModelUtils.loadModel("src/models/my_model.ser");
 * if (loadedModel != null) {
 *     // Use the model for predictions
 * }
 * }</pre>
 *
 * @see MLP
 * @author Brandon Mejia
 * @version 2025-12-03
 */
public class ModelUtils {

    /**
     * Loads a pre-trained MLP model from a file.
     *
     * @param filePath The path to the serialized model file.
     * @return A new {@link MLP} instance with the loaded state, or {@code null} if loading fails.
     */
    public static MLP loadModel(String filePath) {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(filePath))) {
            return (MLP) ois.readObject();
        } catch (IOException | ClassNotFoundException e) {
            System.err.println("ERRO: Falha ao carregar o modelo de " + filePath);
            e.printStackTrace();
            return null;
        }
    }

    /**
     * Saves the current state of an MLP model to a file.
     *
     * @param model    The MLP model to be saved.
     * @param filePath The path to the file where the model will be saved.
     */
    public static void saveModel(MLP model, String filePath) {
        new File(filePath).getParentFile().mkdirs();
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(filePath))) {
            oos.writeObject(model);
        } catch (IOException e) {
            System.err.println("ERRO: Falha ao guardar o modelo em " + filePath);
            e.printStackTrace();
        }
    }

    // ==================================================================================
    //  JSON SERIALIZATION & DESERIALIZATION 
    // ==================================================================================

    /**
     * Saves the current model state (Topology, Activations, Weights, Biases) to a JSON file.
     * @param model    The MLP model to save.
     * @param filePath The destination path for the .json file.
     */
    public static void saveModelToJson(MLP model, String filePath) {
        StringBuilder json = new StringBuilder();
        json.append("{\n");

        // 1. Topology
        Matrix[] w = model.getWeights();
        Matrix[] b = model.getBiases();
        IDifferentiableFunction[] act = model.getActivations();
        int numLayers = w.length + 1;

        int[] topology = new int[numLayers];
        topology[0] = w[0].rows();
        for (int i = 0; i < w.length; i++) {
            topology[i + 1] = w[i].cols();
        }
        json.append("  \"topology\": ").append(Arrays.toString(topology)).append(",\n");

        // 2. Activations
        json.append("  \"activations\": [");
        for (int i = 0; i < act.length; i++) {
            json.append("\"").append(act[i].getClass().getSimpleName()).append("\"");
            if (i < act.length - 1) json.append(", ");
        }
        json.append("],\n");

        // 3. Weights
        json.append("  \"weights\": [");
        for (int i = 0; i < w.length; i++) {
            json.append(matrixToJson(w[i]));
            if (i < w.length - 1) json.append(", ");
        }
        json.append("],\n");

        // 4. Biases
        json.append("  \"biases\": [");
        for (int i = 0; i < b.length; i++) {
            json.append(matrixToJson(b[i]));
            if (i < b.length - 1) json.append(", ");
        }
        json.append("]\n");

        json.append("}");

        try (FileWriter writer = new FileWriter(filePath)) {
            writer.write(json.toString());
            System.out.println("Model saved to JSON: " + filePath);
        } catch (IOException e) {
            System.err.println("Error saving model to JSON: " + e.getMessage());
        }
    }

    /**
     * Loads an MLP model from a JSON file created by {@link #saveModelToJson(MLP, String)}.
     * @param filePath The path to the .json file.
     * @return A new instance of MLP populated with the loaded data.
     */
    public static MLP loadModelFromJson(String filePath) {
        try {
            String content = new String(Files.readAllBytes(Paths.get(filePath)));

            // 1. Extract Topology
            String topoStr = extractJsonValue(content, "topology");
            String[] topoParts = topoStr.replace("[", "").replace("]", "").split(",");
            int[] topology = Arrays.stream(topoParts).map(String::trim).mapToInt(Integer::parseInt).toArray();

            // 2. Extract Activations
            String actStr = extractJsonValue(content, "activations");
            String[] actNames = actStr.replace("[", "").replace("]", "").replace("\"", "").split(",");
            IDifferentiableFunction[] functions = new IDifferentiableFunction[actNames.length];
            for (int i = 0; i < actNames.length; i++) {
                String name = actNames[i].trim();
                if (name.equalsIgnoreCase("Sigmoid")) functions[i] = new Sigmoid();
                else if (name.equalsIgnoreCase("TanH")) functions[i] = new TanH();
                else throw new IllegalArgumentException("Unknown activation function in JSON: " + name);
            }

            // 3. Create MLP Instance
            MLP mlp = new MLP(topology, functions, 1);

            // 4. Extract and Set Weights
            String weightsJson = extractJsonValue(content, "weights");
            mlp.setWeights(parseMatricesFromStream(weightsJson, topology, true));

            // 5. Extract and Set Biases
            String biasesJson = extractJsonValue(content, "biases");
            mlp.setBiases(parseMatricesFromStream(biasesJson, topology, false));

            return mlp;

        } catch (Exception e) {
            throw new RuntimeException("Failed to load MLP from JSON: " + filePath, e);
        }
    }

    // --- JSON Helpers ---

    private static String matrixToJson(Matrix m) {
        StringBuilder sb = new StringBuilder();
        sb.append("[");
        double[][] data = m.getData();
        for (int i = 0; i < data.length; i++) {
            sb.append(Arrays.toString(data[i]));
            if (i < data.length - 1) sb.append(", ");
        }
        sb.append("]");
        return sb.toString();
    }

    private static String extractJsonValue(String json, String key) {
        int keyIndex = json.indexOf("\"" + key + "\"");
        if (keyIndex == -1) throw new IllegalArgumentException("Key not found: " + key);

        int start = json.indexOf(':', keyIndex) + 1;
        while (Character.isWhitespace(json.charAt(start))) start++;

        if (json.charAt(start) == '[') {
            int open = 0;
            for (int i = start; i < json.length(); i++) {
                if (json.charAt(i) == '[') open++;
                if (json.charAt(i) == ']') open--;
                if (open == 0) return json.substring(start, i + 1);
            }
        }
        return "";
    }

    private static Matrix[] parseMatricesFromStream(String jsonArray, int[] topology, boolean isWeights) {
        String clean = jsonArray.replace("[", " ").replace("]", " ").replace(",", " ");
        Scanner scanner = new Scanner(clean);
        scanner.useLocale(java.util.Locale.US);

        int numMatrices = topology.length - 1;
        Matrix[] matrices = new Matrix[numMatrices];

        for (int i = 0; i < numMatrices; i++) {
            int rows = isWeights ? topology[i] : 1;
            int cols = topology[i + 1];
            double[][] data = new double[rows][cols];

            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    if (scanner.hasNextDouble()) {
                        data[r][c] = scanner.nextDouble();
                    }
                }
            }
            matrices[i] = new Matrix(data);
        }
        return matrices;
    }
}