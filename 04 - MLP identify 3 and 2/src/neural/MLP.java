package neural;

import math.Matrix;
import java.io.Serializable;
import neural.activation.IDifferentiableFunction;
import java.util.Random;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicReference;

/**
 * A Multi-Layer Perceptron (MLP) implementation designed for CPU-based computation.
 * <p>
 * This class provides a complete framework for building, training, and evaluating a neural network.
 * It uses a custom {@link math.Matrix} class for all underlying mathematical operations, making it
 * self-contained and independent of external numerical libraries. The core logic includes feedforward
 * prediction and backpropagation with momentum for training.
 * </p>
 *
 * <h3>Advanced Training Features</h3>
 * <p>
 * The {@link #(Matrix, Matrix, Matrix, Matrix, double, int, double)} method implements a sophisticated
 * training loop that includes mini-batch processing, asynchronous validation, best-model checkpointing,
 * and early stopping. This allows for efficient and robust training, preventing overfitting and ensuring
 * the best-performing model is retained.
 * </p>
 *
 * @author hdaniel@ualg.pt, Brandon Mejia
 * @version 202511052038
 */
public class MLP implements Serializable {

    private Matrix[] w;  //weights for each layer
    private Matrix[] b;  //biases for each layer (one per neuron)
    private transient Matrix[] yp; //outputs for each layer (transient)
    private IDifferentiableFunction[] act; //activation functions for each layer
    private int numLayers;
    private transient Matrix[] prevWUpdates; // For momentum (transient)
    private transient Matrix[] prevBUpdates; // For momentum (transient)

    /* Create a Multi-Layer Perceptron with the given layer sizes.
     * layerSizes is an array where each element represents the number of neurons in that layer.
     * For example, new MLP(new int[]{3, 5, 2}) creates a MLP with 3 input neurons,
     * 5 hidden neurons, and 2 output neurons.
     *
     * PRE: layerSizes.length >= 2
     * PRE: act.length == layerSizes.length - 1
     */
    public MLP(int[] layerSizes, IDifferentiableFunction[] act, int seed) {
        if (seed < 0)
            seed = (int) System.currentTimeMillis();

        numLayers = layerSizes.length;

        //setup activation by layer
        this.act = act;

        //create output storage for each layer but the input layer
        yp = new Matrix[numLayers];

        initializeMatrices(layerSizes, seed);
        initializeMomentum();
    }


    // Feed forward propagation
    // also used to predict after training the net
    // yp[0] = X
    // yp[l+1] = Sigmoid( yp[l] * w[l]+b[l] )
    public Matrix predict(Matrix X)
    {
        // Re-initialize transient fields if they are null (e.g., after deserialization)
        if (yp == null) {
            yp = new Matrix[numLayers];
        }
        yp[0] = X;
        for (int l = 0; l < numLayers - 1; l++)
            yp[l + 1] = yp[l].dot(w[l]).addRowVector(b[l]).apply(act[l].fnc());
        return yp[numLayers-1];
    }


    // back propagation
    private Matrix backPropagation(Matrix X, Matrix y, double lr, double momentum, double l2Lambda) {
        Matrix Eout = null;
        Matrix e = null;
        Matrix delta = null;

        // Re-initialize transient fields if they are null
        if (prevWUpdates == null || prevBUpdates == null) {
            initializeMomentum();
        }

        // back propagation using generalised delta rule
        for (int l = numLayers - 2; l >= 0; l--) {
            if (l == numLayers - 2) { // output layer
                Eout = e = y.sub(yp[l + 1]); // e = y – yp[l+1]
            } else { // propagate error
                // e = delta * w[l+1]^T
                e = delta.dot(w[l + 1].transpose());
            }

            // dy = yp[l+1] .* (1-yp[l+1])
            // Note: to compute dy use Sigmoid class derivative
            // in a similar way as in predict()
            Matrix dy = yp[l + 1].apply(act[l].derivative());

            // delta = e .* dy
            delta = e.mult(dy);

            // Calcula a atualização com momentum e regularização L2 (Weight Decay)
            // O gradiente do erro é yp[l]^T * delta
            // O gradiente da regularização L2 é l2Lambda * w[l]
            Matrix grad = yp[l].transpose().dot(delta);
            Matrix wUpdate = grad.add(w[l].mult(l2Lambda)).mult(lr).add(prevWUpdates[l].mult(momentum));
            Matrix bUpdate = delta.sumColumns().mult(lr).add(prevBUpdates[l].mult(momentum));

            w[l] = w[l].add(wUpdate);
            b[l] = b[l].add(bUpdate);

            // Salva as atualizações atuais para a próxima iteração
            prevWUpdates[l] = wUpdate;
            prevBUpdates[l] = bUpdate;
        }
        return Eout;
    }

    private void initializeMatrices(int[] layerSizes, int seed) {
        w = new Matrix[numLayers - 1];
        b = new Matrix[numLayers - 1];

        Random rnd = new Random(seed);
        for (int i = 0; i < numLayers - 1; i++) {
            w[i] = Matrix.Rand(layerSizes[i], layerSizes[i + 1], rnd);
            b[i] = Matrix.Rand(1, layerSizes[i + 1], rnd); // One bias per neuron in the next layer
        }
    }

    private void initializeMomentum() {
        if (w == null) return; // Cannot initialize if weights are not set

        this.prevWUpdates = new Matrix[numLayers - 1];
        this.prevBUpdates = new Matrix[numLayers - 1];
        for (int i = 0; i < numLayers - 1; i++) {
            this.prevWUpdates[i] = new Matrix(w[i].rows(), w[i].cols());
            this.prevBUpdates[i] = new Matrix(b[i].rows(), b[i].cols());
        }
    }

    private Matrix[] cloneMatrices(Matrix[] matrices) {
        if (matrices == null) return null;
        Matrix[] clone = new Matrix[matrices.length];
        for (int i = 0; i < matrices.length; i++) {
            clone[i] = matrices[i].clone();
        }
        return clone;
    }

    /**
     * Trains the MLP model using training and validation datasets, incorporating advanced techniques.
     * <p>
     * This method orchestrates the entire training process, including:
     * <ul>
     *     <li><b>Asynchronous Validation:</b> Performs validation on a separate thread to avoid blocking the training loop.</li>
     *     <li><b>Best Model Checkpointing:</b> Automatically saves the model with the lowest validation error found so far.</li>
     * </ul>
     *
     * @param trainInputs  The matrix of training input data.
     * @param trainOutputs The matrix of training label data.
     * @param valInputs    The matrix of validation input data.
     * @param valOutputs   The matrix of validation label data.
     * @param lr           The learning rate.
     * @param epochs       The total number of epochs to train for.
     * @param momentum     The momentum factor for weight updates.
     * @return The best validation error (MSE) achieved during training.
     */
    public double train(Matrix trainInputs, Matrix trainOutputs, Matrix valInputs, Matrix valOutputs, double lr, int epochs, double momentum, double l2Lambda)
    {
        // --- CONFIGURAÇÕES ---
        int batchSize = 32; // estava em 32
        final int PATIENCE_EPOCHS = 500;
        final int VALIDATION_FREQUENCY = 5;

        // --- INICIALIZAÇÃO ---
        ExecutorService validationExecutor = Executors.newSingleThreadExecutor();
        CompletableFuture<Double> validationFuture = null;

        double bestValidationError = Double.POSITIVE_INFINITY;
        final AtomicReference<MLP> bestMlp = new AtomicReference<>();
        int epochsSinceLastImprovement = 0;
        int totalSamples = trainInputs.rows();
        
        // --- AGENDADOR DE TAXA DE APRENDIZAGEM (LEARNING RATE SCHEDULER) ---
        final double initialLr = lr; // Guarda a taxa de aprendizagem inicial
        // O decayRate controla a velocidade com que a taxa de aprendizagem diminui.
        final double decayRate = initialLr / epochs;

        for (int epoch = 1; epoch <= epochs; epoch++) {
            final double currentLr = initialLr / (1 + decayRate * epoch);
            // --- LOOP DE MINI-BATCHES ---
            for (int i = 0; i < totalSamples; i += batchSize) {
                int end = Math.min(i + batchSize, totalSamples);

                // Fatiar os dados (Slicing) para o lote atual
                Matrix batchX = trainInputs.getRows(i, end);
                Matrix batchY = trainOutputs.getRows(i, end);

                // Treinar apenas neste lote
                this.predict(batchX);
                this.backPropagation(batchX, batchY, currentLr, momentum, l2Lambda);
            }

            // --- VALIDAÇÃO ASSÍNCRONA ---
            if (epoch % VALIDATION_FREQUENCY == 0) {
                if (validationFuture != null) {
                    try {
                        double currentValidationError = validationFuture.get();

                    // Imprime o progresso em intervalos regulares para feedback visual
                    if ((epoch - VALIDATION_FREQUENCY) > 0 && (epoch - VALIDATION_FREQUENCY) % 100 == 0) {
                        System.out.printf("Época: %-5d | LR: %.8f | (MSE): %.6f\n", epoch, currentLr, currentValidationError);
                    }

                        // Verifica se o modelo melhorou
                        if (currentValidationError < bestValidationError) {
                            bestValidationError = currentValidationError;
                            bestMlp.set(this.clone()); // Checkpoint tipo um jogo
                            epochsSinceLastImprovement = 0;
                        } else {
                            epochsSinceLastImprovement += VALIDATION_FREQUENCY; // Incrementa pelo intervalo
                        }

                        // Checagem de Parada Antecipada
                        if (epochsSinceLastImprovement >= PATIENCE_EPOCHS) {
                            System.out.printf("\nParada antecipada na época %d. Sem melhoria há %d épocas. Melhor erro: %.6f\n", epoch, epochsSinceLastImprovement, bestValidationError);
                            validationFuture.cancel(true); // Cancela a próxima validação
                            break;
                        }

                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }
                // Lança a próxima validação em background
                final MLP modelCloneForValidation = this.clone();
                validationFuture = CompletableFuture.supplyAsync(() -> {
                    Matrix valPrediction = modelCloneForValidation.predict(valInputs);
                    return valOutputs.sub(valPrediction).apply(x -> x * x).sum() / valInputs.rows();
                }, validationExecutor);
            }
        }

        validationExecutor.shutdown();

        if (bestMlp.get() != null) {
            this.setWeights(bestMlp.get().getWeights());
            this.setBiases(bestMlp.get().getBiases());
        }

        return bestValidationError;
    }

    /**
     * Returns the learned weight matrices for each layer.
     * @return An array of Matrix objects representing the weights.
     */
    public Matrix[] getWeights() {
        return cloneMatrices(this.w);
    }

    /**
     * Returns the learned bias values for each layer.
     * @return An array of doubles representing the biases.
     */
    public Matrix[] getBiases() {
        return cloneMatrices(this.b);
    }

    /**
     * Returns the activation functions used in each layer.
     * @return An array of IDifferentiableFunction.
     */
    public IDifferentiableFunction[] getActivations() {
        return this.act;
    }

    /**
     * Sets the weight matrices for each layer.
     * <p>
     * This method allows for manually setting the weights of the network, for example,
     * to test a pre-calculated configuration without training.
     *
     * @param newWeights An array of Matrix objects representing the new weights.
     * @throws IllegalArgumentException if the number of matrices or their dimensions do not match the network's topology.
     */
    public void setWeights(Matrix[] newWeights)
    {
        if (newWeights.length != this.w.length) {
            throw new IllegalArgumentException("Invalid number of weight matrices. Expected: " + this.w.length + ", Got: " + newWeights.length);
        }
        for (int i = 0; i < newWeights.length; i++) {
            if (newWeights[i].rows() != this.w[i].rows() || newWeights[i].cols() != this.w[i].cols()) {
                throw new IllegalArgumentException("Incompatible dimensions for weight matrix at layer " + i +
                        ". Expected: " + this.w[i].rows() + "x" + this.w[i].cols() +
                        ", Got: " + newWeights[i].rows() + "x" + newWeights[i].cols());
            }
        }
        this.w = cloneMatrices(newWeights);
    }

    public void setBiases(Matrix[] newBiases) {
        if (newBiases.length != this.b.length) {
            throw new IllegalArgumentException("Invalid number of bias matrices. Expected: " + this.b.length + ", Got: " + newBiases.length);
        }
        for (int i = 0; i < newBiases.length; i++) {
            if (newBiases[i].rows() != this.b[i].rows() || newBiases[i].cols() != this.b[i].cols()) {
                throw new IllegalArgumentException("Incompatible dimensions for bias matrix at layer " + i +
                        ". Expected: " + this.b[i].rows() + "x" + this.b[i].cols() +
                        ", Got: " + newBiases[i].rows() + "x" + newBiases[i].cols());
            }
        }
        this.b = cloneMatrices(newBiases);
    }

    // Overload for compatibility with single-neuron test case
    public void setBiases(double[] newBiases)
    {
        if (newBiases.length == this.b.length) {
            for (int i = 0; i < newBiases.length; i++)
            {
                if (this.b[i].cols() == 1) {
                    this.b[i] = new Matrix(new double[][]{{newBiases[i]}});
                } else {
                    throw new IllegalArgumentException("Cannot set single double bias for a layer with multiple neurons.");
                }
            }
        } else {
            throw new IllegalArgumentException("Invalid number of biases. Expected: " + this.b.length + ", Got: " + newBiases.length);
        }
    }

    /**
     * Creates a deep copy of the current MLP instance.
     * The new MLP will have the same topology, activation functions, weights, and biases,
     * but it will be a completely independent object.
     *
     * @return A new MLP instance that is a clone of the current one.
     */
    public MLP clone() {
        // 1. Recreate the topology from the weight matrices
        int[] topology = new int[this.numLayers];
        topology[0] = this.w[0].rows();
        for (int i = 0; i < this.w.length; i++) {
            topology[i + 1] = this.w[i].cols();
        }

        // 2. Create a new MLP instance with the same topology and activation functions
        // The seed is not relevant here, as we are going to replace the weights and biases.
        MLP clonedMlp = new MLP(topology, this.act, 1);

        // 3. Copy the weights and biases
        clonedMlp.setWeights(this.getWeights());
        clonedMlp.setBiases(this.getBiases());

        return clonedMlp;
    }

}
