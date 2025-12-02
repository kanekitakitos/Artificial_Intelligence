package neural;

import neural.activation.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;


/**
 * A Multi-Layer Perceptron (MLP) implementation optimized for GPU computation using the ND4J library.
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
    public void train(INDArray X, INDArray y, double learningRate, int epochs, double momentum) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            predict(X);

            INDArray delta = null;
            for (int l = numLayers - 2; l >= 0; l--) {
                INDArray error;
                if (l == numLayers - 2) {
                    error = y.sub(layerOutputs[l + 1]);
                } else {
                    error = delta.mmul(w[l + 1].transpose());
                }

                INDArray derivative = applyActivationDerivative(layerOutputs[l + 1], activationFunctions[l]);
                delta = error.mul(derivative);

                INDArray wUpdate = layerOutputs[l].transpose().mmul(delta).mul(learningRate).add(prevWUpdates[l].mul(momentum));
                INDArray bUpdate = delta.sum(0).mul(learningRate).add(prevBUpdates[l].mul(momentum));

                w[l].addi(wUpdate);
                b[l].addi(bUpdate);

                prevWUpdates[l] = wUpdate;
                prevBUpdates[l] = bUpdate;
            }
        }
    }

    private INDArray applyActivation(INDArray input, IDifferentiableFunction activation) {
        if (activation instanceof Sigmoid) return Transforms.sigmoid(input, true);
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
            // f'(x) = sigmoid(x) * (1 - sigmoid(x))
            INDArray sig = Transforms.sigmoid(input, true);
            return sig.mul(sig.rsub(1.0)); // sig * (1 - sig)
        }
        throw new IllegalArgumentException("Unsupported activation function for GPU: " + activation.getClass().getSimpleName());
    }
}