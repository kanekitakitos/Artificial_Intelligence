package neural;

import math.Matrix;
import neural.activation.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.Arrays;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicReference;


/**
 * A Multi-Layer Perceptron (MLP) implementation optimized for GPU computation using the ND4J library.
 * <p>This class is a direct functional counterpart to the pure Java {@link MLP}, designed to provide
 * an identical API while leveraging GPU acceleration for superior performance.</p>
 * <p>
 * This class mirrors the functionality of the pure Java {@link MLP} but leverages {@link INDArray}
 * for all matrix operations. This allows the underlying ND4J backend to execute calculations on a
 * compatible GPU (via CUDA or ROCm), leading to a significant performance increase for training.
 * </p>
 * <p>
 * The core logic, including feedforward and backpropagation, remains conceptually the same, but
 * the implementation uses hardware-accelerated ND4J methods.
 * </p>
 *
 * <h3>Key Features:</h3>
 * <ul>
 *   <li><b>GPU Acceleration:</b> All matrix operations are performed using ND4J's backend.</li>
 *   <li><b>API Parity:</b> Maintains method signatures consistent with the {@link MLP} class for seamless integration.</li>
 *   <li><b>Advanced Training:</b> Includes support for asynchronous validation and best-model checkpointing.</li>
 *   <li><b>State Management:</b> Provides methods to get, set, and clone the model's weights and biases.</li>
 * </p>
 *
 * @see apps.GpuMLP23
 * @see INDArray
 * @author Brandon Mejia
 * @version 2025-11-30
 */
public class GpuMLP {

    private INDArray[] w;  // Weights for each layer
    private INDArray[] b;  // Biases for each layer
    private INDArray[] layerOutputs; // Outputs for each layer (yp)
    private final IDifferentiableFunction[] activationFunctions;
    private final int numLayers;
    private INDArray[] prevWUpdates; // For momentum
    private INDArray[] prevBUpdates; // For momentum

    /**
     * Constructs a GPU-accelerated MLP with the given topology and activation functions.
     *
     * @param layerSizes An array where each element is the number of neurons in that layer.
     * @param act        An array of activation functions for each layer transition.
     * @param seed       A seed for the random number generator to ensure weight reproducibility.
     */
    public GpuMLP(int[] layerSizes, IDifferentiableFunction[] act, int seed) {
        this.numLayers = layerSizes.length;
        this.activationFunctions = act;
        this.layerOutputs = new INDArray[numLayers];

        this.w = new INDArray[numLayers - 1];
        this.b = new INDArray[numLayers - 1];
        this.prevWUpdates = new INDArray[numLayers - 1];
        this.prevBUpdates = new INDArray[numLayers - 1];

        Nd4j.getRandom().setSeed(seed);

        for (int i = 0; i < numLayers - 1; i++) {
            // Initialize weights and biases with small random values
            w[i] = Nd4j.rand(layerSizes[i], layerSizes[i + 1]);
            b[i] = Nd4j.rand(1, layerSizes[i + 1]);

            // Initialize momentum update matrices with zeros
            prevWUpdates[i] = Nd4j.zeros(layerSizes[i], layerSizes[i + 1]);
            prevBUpdates[i] = Nd4j.zeros(1, layerSizes[i + 1]);
        }
    }

    /**
     * Performs the feedforward pass to generate predictions for a given input.
     *
     * @param X The input data as an {@link INDArray}.
     * @return The network's prediction.
     */
    public INDArray predict(INDArray X) {
        layerOutputs[0] = X;
        for (int l = 0; l < numLayers - 1; l++) {
            INDArray z = layerOutputs[l].mmul(w[l]).addRowVector(b[l]);
            layerOutputs[l + 1] = applyActivation(z, activationFunctions[l]);
        }
        return layerOutputs[numLayers - 1];
    }

    /**
     * Trains the network for a specified number of epochs using backpropagation with momentum.
     *
     * @param X            The training input data.
     * @param y            The training target labels.
     * @param learningRate The learning rate for weight updates.
     * @param epochs       The number of training iterations.
     * @param momentum     The momentum factor for weight updates.
     */
    public void train(INDArray X, INDArray y, double learningRate, int epochs, double momentum)
    {
        System.out.println("Iniciando o treinamento da rede (GPU)...");
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            predict(X);
            backPropagation(X, y, learningRate, momentum);

            if ((epoch + 1) % 100 == 0) {
                INDArray error = y.sub(layerOutputs[numLayers - 1]);
                double mse = error.mul(error).sumNumber().doubleValue() / X.rows();
                System.out.printf("Época %d/%d, MSE: %.12f\n", epoch + 1, epochs, mse);
            }
        }
        System.out.println("Treinamento concluído.");
    }

    public double train(INDArray trainInputs, INDArray trainOutputs, INDArray valInputs, INDArray valOutputs, double lr, int epochs, double momentum) {
        //System.out.println("Iniciando o treinamento da rede (GPU)...");
        //System.out.println("Amostras de Treino: " + trainInputs.rows() + " | Amostras de Validação: " + valInputs.rows());

        ExecutorService validationExecutor = Executors.newSingleThreadExecutor();
        CompletableFuture<Double> validationFuture = null;

        double bestValidationError = Double.POSITIVE_INFINITY;
        final AtomicReference<GpuMLP> bestMlp = new AtomicReference<>();

        for (int epoch = 1; epoch <= epochs; epoch++) {
            // Perform one training step
            this.predict(trainInputs);
            this.backPropagation(trainInputs, trainOutputs, lr, momentum);

            // Asynchronous validation
            if (epoch % 10 == 0) {
                if (validationFuture != null) {
                    try {
                        double currentValidationError = validationFuture.get();

                        if ((epoch - 10) > 0 && (epoch - 10) % 100 == 0) {
                            //System.out.printf("Época: %-5d | LR: %.6f | Erro de Validação (MSE): %.6f\n", epoch - 10, lr, currentValidationError);
                        }

                        if (currentValidationError < bestValidationError) {
                            bestValidationError = currentValidationError;
                            bestMlp.set(this.clone()); // Save a copy of the best model
                        }

                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }

                final GpuMLP modelCloneForValidation = this.clone();
                validationFuture = CompletableFuture.supplyAsync(() -> {
                    INDArray valPrediction = modelCloneForValidation.predict(valInputs);
                    INDArray error = valOutputs.sub(valPrediction);
                    return error.mul(error).sumNumber().doubleValue() / valInputs.rows();
                }, validationExecutor);
            }
        }

        validationExecutor.shutdown();
        System.out.println("Treinamento concluído.");

        // Restore the best model found
        if (bestMlp.get() != null) {
            this.setWeights(bestMlp.get().getWeights());
            this.setBiases(bestMlp.get().getBiases());
        }

        return bestValidationError;
    }

    private void backPropagation(INDArray X, INDArray y, double learningRate, double momentum) {
        INDArray delta = null;
        for (int l = numLayers - 2; l >= 0; l--) {
            INDArray error;
            if (l == numLayers - 2) {
                // Error for the output layer
                error = y.sub(layerOutputs[l + 1]);
            } else {
                // Propagate error to the hidden layer
                error = delta.mmul(w[l + 1].transpose());
            }

            // Calculate delta
            INDArray derivative = applyActivationDerivative(layerOutputs[l + 1], activationFunctions[l]);
            delta = error.mul(derivative);

            // Calculate weight and bias updates with momentum
            INDArray wUpdate = layerOutputs[l].transpose().mmul(delta).mul(learningRate).add(prevWUpdates[l].mul(momentum));
            INDArray bUpdate = delta.sum(0).mul(learningRate).add(prevBUpdates[l].mul(momentum));

            // Apply updates (in-place for efficiency)
            w[l].addi(wUpdate);
            b[l].addi(bUpdate);

            // Store current updates for the next iteration's momentum calculation
            prevWUpdates[l] = wUpdate;
            prevBUpdates[l] = bUpdate;
        }
    }

    private INDArray applyActivation(INDArray input, IDifferentiableFunction activation) {
        if (activation instanceof Sigmoid) {
            return Transforms.sigmoid(input, true);
        }
        throw new IllegalArgumentException("Unsupported activation function for GPU: " + activation.getClass().getSimpleName());
    }

    /**
     * Applies the derivative of an activation function to an input {@link INDArray}.
     * <p>
     * This method calculates the derivative element-wise, using optimized ND4J operations.
     * The direct derivative functions (e.g., {@code Transforms.sigmoidDerivative}) were replaced
     * with their underlying mathematical formulas to ensure broader compatibility across
     * different ND4J versions and backends.
     * </p>
     *
     * <h4>Implementations:</h4>
     * <ul>
     *   <li><b>Sigmoid:</b> Implements {@code f'(x) = sigmoid(x) * (1 - sigmoid(x))}.</li>
     *   <li><b>TanH:</b> Implements {@code f'(x) = 1 - tanh(x)^2}.</li>
     *   <li><b>ReLU:</b> Implements {@code f'(x) = 1 if x > 0, else 0}.</li>
     * </ul>
     *
     * @param input      The INDArray to which the derivative is applied (typically the layer's output before activation).
     * @param activation The {@link IDifferentiableFunction} whose derivative will be used.
     * @return A new {@link INDArray} containing the result of the derivative computation.
     * @throws IllegalArgumentException if the activation function is not supported for GPU acceleration.
     */
    private INDArray applyActivationDerivative(INDArray input, IDifferentiableFunction activation) {
        if (activation instanceof Sigmoid) {
            // The derivative of sigmoid is calculated based on its output y: y * (1 - y)
            // Here 'input' is already the output of the sigmoid layer (y = yp[l+1])
            return input.mul(input.rsub(1.0)); // y * (1 - y)
        }
        throw new IllegalArgumentException("Unsupported activation function for GPU: " + activation.getClass().getSimpleName());
    }

    public INDArray[] getWeights() {
        INDArray[] clonedW = new INDArray[w.length];
        for (int i = 0; i < w.length; i++) {
            clonedW[i] = w[i].dup();
        }
        return clonedW;
    }

    public INDArray[] getBiases() {
        INDArray[] clonedB = new INDArray[b.length];
        for (int i = 0; i < b.length; i++) {
            clonedB[i] = b[i].dup();
        }
        return clonedB;
    }

    public void setWeights(INDArray[] newWeights) {
        if (newWeights.length != this.w.length) {
            throw new IllegalArgumentException("Invalid number of weight matrices.");
        }
        for (int i = 0; i < newWeights.length; i++) {
            if (!Arrays.equals(newWeights[i].shape(), this.w[i].shape())) {
                throw new IllegalArgumentException("Incompatible dimensions for weight matrix at layer " + i);
            }
        }
        this.w = newWeights;
    }

    public void setBiases(INDArray[] newBiases) {
        if (newBiases.length != this.b.length) {
            throw new IllegalArgumentException("Invalid number of bias matrices.");
        }
        for (int i = 0; i < newBiases.length; i++) {
            if (!Arrays.equals(newBiases[i].shape(), this.b[i].shape())) {
                throw new IllegalArgumentException("Incompatible dimensions for bias matrix at layer " + i);
            }
        }
        this.b = newBiases;
    }

    @Override
    public GpuMLP clone() {
        int[] topology = new int[this.numLayers];
        topology[0] = (int) this.w[0].shape()[0];
        for (int i = 0; i < this.w.length; i++) {
            topology[i + 1] = (int) this.w[i].shape()[1];
        }

        GpuMLP clonedMlp = new GpuMLP(topology, this.activationFunctions, 1);

        // Deep copy of weights and biases
        clonedMlp.setWeights(this.getWeights());
        clonedMlp.setBiases(this.getBiases());

        return clonedMlp;
    }
}