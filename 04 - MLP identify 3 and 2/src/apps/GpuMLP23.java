package apps;

import neural.GpuMLP;
import math.Matrix;
import neural.activation.IDifferentiableFunction;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 * A high-level trainer for a GPU-accelerated MLP, designed for hyperparameter tuning.
 * <p>
 * This class serves as a wrapper around the {@link GpuMLP}, managing the data conversion
 * from standard Java arrays to ND4J's {@link INDArray} format and orchestrating the training process.
 * It is specifically designed to be called by the {@link HyperparameterTuner} to run a single,
 * fast training trial on the GPU.
 * <br>
 * Unlike {@code MLP23}, this class does not implement complex logic like early stopping or
 * adaptive learning rates. Its sole purpose is to provide a standardized, high-speed training
 * execution for the grid search process.
 * </p>
 *
 * @see GpuMLP
 * @see HyperparameterTuner
 * @author Brandon Mejia
 * @version 2025-11-30
 */
public class GpuMLP23 {

    private final GpuMLP mlp;
    private final double lr;
    private final int epochs;
    private final double momentum;
    private static final int SEED = 4;

    public GpuMLP23(int[] topology, IDifferentiableFunction[] functions, double lr, double momentum, int epochs) {
        this.lr = lr;
        this.momentum = momentum;
        this.epochs = epochs;
        this.mlp = new GpuMLP(topology, functions, SEED);
    }

    /**
     * Executes the training process on the GPU.
     *
     * <p>This method orchestrates a single, complete training trial. It performs the following steps:</p>
     * <ol>
     *   <li>Loads the training data from the specified file paths using the {@link DataHandler}.</li>
     *   <li>Converts the data from the project's {@code Matrix} format to ND4J's {@code INDArray} format to enable GPU computation.</li>
     *   <li>Initiates the training loop on the underlying {@link GpuMLP} instance, which performs the backpropagation and weight updates.</li>
     * </ol>
     */
    public void train() {
        // Load data using the existing DataHandler
        DataHandler dataHandler = new DataHandler(SEED); // No validation split needed for this trainer
        Matrix trainInputsMatrix = dataHandler.getTrainInputs();
        Matrix trainOutputsMatrix = dataHandler.getTrainOutputs();

        // Convert data to ND4J INDArrays
        INDArray trainInputs = Nd4j.create(trainInputsMatrix.getData()).castTo(Nd4j.dataType());
        INDArray trainOutputs = Nd4j.create(trainOutputsMatrix.getData()).castTo(Nd4j.dataType());

        // Train the GPU-enabled MLP
        mlp.train(trainInputs, trainOutputs, this.lr, this.epochs, this.momentum);
    }

    /**
     * Executes the training process on the GPU using pre-loaded data.
     * <p>This overload is optimized for hyperparameter tuning, as it avoids reloading
     * data from disk on every trial.</p>
     *
     * @param trainInputsMatrix The training input data as a {@link Matrix}.
     * @param trainOutputsMatrix The training output data as a {@link Matrix}.
     */
    public void train(Matrix trainInputsMatrix, Matrix trainOutputsMatrix) {
        // Convert data to ND4J INDArrays
        INDArray trainInputs = Nd4j.create(trainInputsMatrix.getData()).castTo(Nd4j.dataType());
        INDArray trainOutputs = Nd4j.create(trainOutputsMatrix.getData()).castTo(Nd4j.dataType());

        mlp.train(trainInputs, trainOutputs, this.lr, this.epochs, this.momentum);
    }

    /**
     * Evaluates the trained network against a test dataset and calculates its accuracy.
     *
     * @return The accuracy of the model as a percentage (e.g., 97.5).
     */
    public double test() {
        // Load the default test data directly into Matrix objects.
        Matrix[] testData = DataHandler.loadDefaultTestData();

        return test(testData[0], testData[1]);
    }

    /**
     * Evaluates the trained network against a pre-loaded test dataset.
     * <p>This overload is optimized for hyperparameter tuning.</p>
     *
     * @param testInputsMatrix The test input data as a {@link Matrix}.
     * @param expectedOutputsMatrix The expected output data as a {@link Matrix}.
     * @return The accuracy of the model as a percentage.
     */
    public double test(Matrix testInputsMatrix, Matrix expectedOutputsMatrix) {
        // Convert to INDArray format
        INDArray testInputs = Nd4j.create(testInputsMatrix.getData()).castTo(Nd4j.dataType());
        INDArray expectedOutputs = Nd4j.create(expectedOutputsMatrix.getData()).castTo(Nd4j.dataType());

        // Get predictions from the trained MLP
        INDArray predictedOutputs = mlp.predict(testInputs);

        // Round predictions to 0 or 1 for classification
        INDArray roundedPredictions = Transforms.round(predictedOutputs);

        // Compare predictions with expected labels to calculate accuracy
        long correctPredictions = 0;
        long totalPredictions = expectedOutputs.rows();

        for (int i = 0; i < totalPredictions; i++) {
            if (roundedPredictions.getDouble(i, 0) == expectedOutputs.getDouble(i, 0)) {
                correctPredictions++;
            }
        }

        // Return accuracy as a percentage
        return (double) correctPredictions / totalPredictions * 100.0;
    }
}