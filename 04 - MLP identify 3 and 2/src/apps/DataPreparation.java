package apps;

/**
 * A utility class for performing one-off data preparation tasks.
 * <p>
 * This class is designed to be run as a script to preprocess and organize datasets before they are used for training
 * or testing machine learning models. Its primary function is to concatenate multiple smaller CSV files into
 * larger, consolidated files, which can then be used as a complete training or labeling set.
 * <p>
 * It uses the static methods provided by the {@link DataHandler} class to perform the file operations.
 *
 * <h3>Example Usage</h3>
 * <p>
 * To use this script, configure the {@code inputFilesToConcatenate} and {@code labelFilesToConcatenate} arrays
 * with the paths to the files you wish to merge. Then, uncomment the calls to {@code DataHandler.concatenateAndSaveCsv}
 * to execute the concatenation.
 * </p>
 *
 * <pre>{@code
 * // 1. Define the input and label files to be merged.
 * String[] inputFiles = {"src/data/dataset_part1.csv", "src/data/dataset_part2.csv"};
 * String[] labelFiles = {"src/data/labels_part1.csv", "src/data/labels_part2.csv"};
 *
 * // 2. Concatenate and save the merged files.
 * DataHandler.concatenateAndSaveCsv(inputFiles, "src/data/treino_inputs_concatenados.csv");
 * DataHandler.concatenateAndSaveCsv(labelFiles, "src/data/treino_labels_concatenados.csv");
 * }</pre>
 *
 * @see DataHandler
 * @author Brandon Mejia
 * @version 2025-11-29
 */
public class DataPreparation {

    public static void main(String[] args) {
        System.out.println("--- A iniciar a preparação dos dados ---");

        // 1. Definir os ficheiros de entrada e saída que quer juntar
        String[] inputFilesToConcatenate = {
                "src/data/dataset.csv",
                //"src/data/dataset_novos.csv",
                //"src/data/dataset_apenas_novos.csv",
                //"src/data/dataset_apenas_novos2.csv"
        };

        String[] labelFilesToConcatenate = {
                "src/data/labels.csv",
                //"src/data/labels.csv", // Repetido para corresponder aos inputs
                //"src/data/labels.csv", // Repetido para corresponder aos inputs
                //"src/data/labels.csv"  // Repetido para corresponder aos inputs
        };

        // 2. Chamar o método para concatenar e guardar os ficheiros
        //DataHandler.concatenateAndSaveCsv(inputFilesToConcatenate, "src/data/treino_inputs_concatenados.csv");
        //DataHandler.concatenateAndSaveCsv(labelFilesToConcatenate, "src/data/treino_labels_concatenados.csv");

        System.out.println("\n--- Preparação de dados concluída ---");
    }
}