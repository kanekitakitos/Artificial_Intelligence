package apps;

import math.Matrix;
import neural.activation.IDifferentiableFunction;
import neural.MLP;
import neural.activation.Sigmoid;
import neural.activation.Step;
import java.util.Scanner;

/**
 * @author Brandon Mejia
 * @version 25/11/2025
 */
public class MLPNXOR {

    public static void main(String[] args)
    {
        boolean single = false;

        if(single)
            singleNeuron();
        else
            mlp();




    }

    /**
     * Imprime os pesos e biases de uma MLP de forma detalhada, neurónio a neurónio.
     * @param title O título a ser exibido antes da impressão.
     * @param mlp A instância da MLP cujos pesos serão impressos.
     */
    private static void printWeightsAndBiases(String title, MLP mlp) {
        System.out.println(title);
        Matrix[] weights = mlp.getWeights();
        Matrix[] biases = mlp.getBiases();

        // Camada Oculta (Input -> Hidden)
        System.out.println("\n[Camada Oculta]");
        Matrix hiddenWeights = weights[0];
        Matrix hiddenBiases = biases[0];
        for (int j = 0; j < hiddenWeights.cols(); j++) { // Itera por cada neurónio da camada oculta
            System.out.printf("Neurónio Oculto %d:\n", j);
            System.out.printf("  - Bias (w0):   % .4f\n", hiddenBiases.get(0, j));
            for (int i = 0; i < hiddenWeights.rows(); i++) { // Itera pelas conexões de entrada
                System.out.printf("  - Peso da Entrada %d (w%d): % .4f\n", i, i + 1, hiddenWeights.get(i, j));
            }
        }

        // Camada de Saída (Hidden -> Output)
        System.out.println("\n[Camada de Saída]");
        Matrix outputWeights = weights[1];
        Matrix outputBiases = biases[1];
        System.out.println("Neurónio de Saída 0:");
        System.out.printf("  - Bias (w0):   % .4f\n", outputBiases.get(0, 0));
        System.out.printf("  - Peso do Neurónio Oculto 0 (w1): % .4f\n", outputWeights.get(0, 0));
        System.out.printf("  - Peso do Neurónio Oculto 1 (w2): % .4f\n", outputWeights.get(1, 0));

        System.out.println("----------------------------");
    }


    static public void mlp()
    {
        double lr    = 0.4;  // learning rate 0 > lr > 1
        int    epochs = 10000; //define the number of epochs in the order of thousands
        int[] topology = {2, 2, 1};

        //Dataset
        Matrix trX = new Matrix(
                new double[][]{
                        {0, 0},
                        {0, 1},
                        {1, 0},
                        {1, 1}});

        Matrix trY = new Matrix(
                new double[][]{
                        {1},
                        {0},
                        {0},
                        {1}});


        //Train the MLP
        MLP mlp = new MLP(topology,
                new IDifferentiableFunction[]{
                        new Sigmoid(),
                        new Sigmoid(),},
                -1);

        // --- Treinar a rede ---
        // Como o método train agora requer dados de validação, podemos simplesmente
        // reutilizar os dados de treino para essa finalidade, já que este é um exemplo simples.
        // Também definimos valores padrão para momentum e l2Lambda.
        double momentum = 0.9;
        double l2Lambda = 0.0; // Sem regularização para este exemplo
        mlp.train(trX, trY, trX, trY, lr, epochs, momentum, l2Lambda);


        //Get input and create evaluation Matrix
        Scanner sc = new Scanner(System.in);
        int n = sc.nextInt();
        double[][] input = new double[n][2];
        for (int i = 0; i < n; i++) {
            input[i][0] = Double.parseDouble(sc.next().replace(',', '.'));
            input[i][1] = Double.parseDouble(sc.next().replace(',', '.'));
        }
        Matrix evX = new Matrix(input);


        // Imprimir os pesos finais, após o treino
        printWeightsAndBiases("\n--- Pesos Finais por Neurónio ---", mlp);

        //Predict and output results
        Matrix pred = mlp.predict(evX);

        //convert probabilities to integer classes: 0 or 1
        pred = pred.apply(new Step().fnc());

        //print output
        for (int i = 0; i < pred.rows(); i++) {
            System.out.printf("%d\n", (int) pred.get(i, 0));
        }

        sc.close();
    }

    public static void singleNeuron()
    {
        // 1. Define o dataset para a lógica y = x2 (padrão 0101)
        double[][] inputs = {
                {0, 0},
                {0, 1},
                {1, 0},
                {1, 1}
        };
        double[] expectedOutputs = {0, 1, 0, 1};

        // 2. Define os pesos e o bias calculados manualmente
        // Para y = x2, podemos usar pesos que focam em x2 e ignoram x1.
        // A lógica é: 0*x1 + 1*x2 - 0.5 > 0  =>  x2 > 0.5
        Matrix weights = new Matrix(new double[][]{{0.0}, {1.0}}); // Shape: 2x1
        double bias = -0.5;

        // 3. Cria e configura uma MLP para atuar como um único neurónio
        int[] topology = {2, 1}; // 2 entradas, 1 neurónio de saída
        // Nota: Step não é diferenciável, mas é usada aqui como função de ativação para predição.
        IDifferentiableFunction[] activations = {new Step()};
        MLP mlp = new MLP(topology, activations, -1);

        // Usa setters para aplicar os pesos e bias calculados manualmente
        mlp.setWeights(new Matrix[]{weights});
        //mlp.setBiases(new double[]{bias});

        System.out.println("Testando uma MLP como um único neurónio para a lógica y = x2 (padrão 0101):");
        System.out.println("Pesos: w1=0.0, w2=1.0, Bias: -0.5");
        System.out.println("-----------------------------------------");

        // Testa com o dataset predefinido
        Matrix testMatrix = new Matrix(inputs);
        Matrix predictions = mlp.predict(testMatrix);
        for (int i = 0; i < inputs.length; i++) {
            double prediction = predictions.get(i, 0);
            System.out.printf("Input: [%.0f, %.0f] -> Predicted: %.0f, Expected: %.0f\n",
                    inputs[i][0], inputs[i][1], prediction,
                    expectedOutputs[i]);
        }

        // --- Parte de Teste Interativo ---
        System.out.println("\n--- Modo de Teste Interativo ---");
        System.out.println("Digite o número de pares a testar, seguido pelos pares (ex: 1.0 0.9):");
        Scanner sc = new Scanner(System.in);
        int n = 0;
        try {
            n = sc.nextInt();
        } catch (java.util.InputMismatchException e) {
            System.err.println("\nErro: Entrada inválida. Por favor, insira um número inteiro para a quantidade de pares.");
            sc.close();
            return; // Sai do método se a primeira entrada for inválida
        }

        for (int i = 0; i < n; i++) {
            try {
                // Cria um array para os dois valores de entrada
                double[][] testInput = new double[1][2];
                testInput[0][0] = Double.parseDouble(sc.next().replace(',', '.'));
                testInput[0][1] = Double.parseDouble(sc.next().replace(',', '.'));

                // Obtém a predição do neurónio
                double prediction = mlp.predict(new Matrix(testInput)).get(0, 0);
                System.out.printf("Input: [%.1f, %.1f] -> Predicted: %d\n", testInput[0][0], testInput[0][1], (int) prediction);
            } catch (NumberFormatException e) {
                System.err.println("Erro: Entrada de par inválida. Por favor, insira números (ex: 0.0 1.0). Pulando este par.");
                // Limpa a linha atual do scanner para evitar erros em cascata
                if (sc.hasNextLine()) {
                    sc.nextLine();
                }
            }
        }
        sc.close();
    }

}
