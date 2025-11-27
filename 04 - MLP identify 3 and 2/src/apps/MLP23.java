package apps;

import math.Matrix;
import neural.activation.IDifferentiableFunction;
import neural.activation.*;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.Scanner;
import neural.MLP;

public class MLP23 {

    private double lr = 0.006;

    private int epochs = 5000;
    private int[] topology = {400,4, 1};
    private IDifferentiableFunction[] functions = {new Sigmoid(),new Sigmoid(), new Sigmoid()};
    private MLP mlp;
    private static int seek = 4;

    private static class DataPoint {
        final double[] input;
        final double[] output;
        DataPoint(double[] input, double[] output) { this.input = input; this.output = output; }
    }

    public MLP23() {
        this.mlp = new MLP(topology,
               functions, seek);
    }


    private List<double[]> loadData(String filePath) {
        List<double[]> data = new ArrayList<>();
        String line;
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            while ((line = br.readLine()) != null) {
                String[] stringValues = line.split(",");
                double[] doubleValues = new double[stringValues.length];
                for (int i = 0; i < stringValues.length; i++) {
                    doubleValues[i] = Double.parseDouble(stringValues[i]);
                }
                data.add(doubleValues);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return data;
    }

    public void train(String[] inputPaths, String[] outputPaths) {
        if (inputPaths.length != outputPaths.length) {
            throw new IllegalArgumentException("La cantidad de archivos de entrada debe ser igual a la cantidad de archivos de salida.");
        }

        List<DataPoint> combinedData = new ArrayList<>();

        for (int fileIndex = 0; fileIndex < inputPaths.length; fileIndex++) {
            List<double[]> inputDataList = loadData(inputPaths[fileIndex]);
            List<double[]> outputDataList = loadData(outputPaths[fileIndex]);

            if (inputDataList.size() != outputDataList.size()) {
                throw new IllegalStateException("El archivo de entrada " + inputPaths[fileIndex] + " y el de salida " + outputPaths[fileIndex] + " no tienen el mismo número de líneas.");
            }

            for (int i = 0; i < inputDataList.size(); i++) {
                double[] currentInput = inputDataList.get(i);
                // Normalizar a entrada para o intervalo [0, 1] dividindo por 255.0
                for (int j = 0; j < currentInput.length; j++) {
                    currentInput[j] /= 255.0;
                }

                double[] currentOutput = outputDataList.get(i);
                if (currentOutput[0] == 2.0) {
                    currentOutput[0] = 0.0;
                } else if (currentOutput[0] == 3.0) {
                    currentOutput[0] = 1.0;
                }
                combinedData.add(new DataPoint(currentInput, currentOutput));
            }
        }

        //Collections.shuffle(combinedData, new Random(seek));
        Collections.shuffle(combinedData, new Random(seek));
        double[][] shuffledInputs = new double[combinedData.size()][];
        double[][] shuffledOutputs = new double[combinedData.size()][];
        for (int i = 0; i < combinedData.size(); i++) {
            shuffledInputs[i] = combinedData.get(i).input;
            shuffledOutputs[i] = combinedData.get(i).output;
        }

        System.out.println("Iniciando o treinamento da rede...");
        System.out.println("Amostras: " + shuffledInputs.length + " | LR: " + this.lr);

        // Treino
        mlp.train(new Matrix(shuffledInputs), new Matrix(shuffledOutputs), this.lr, this.epochs);
        System.out.println("Treinamento concluído.");

        // Podes salvar os pesos aqui se quiseres
    }

    public MLP getMLP() { return this.mlp; }

    public static void main(String[] args) {
        // Passa uma seed para garantir reprodutibilidade
        MLP23 model = new MLP23();


        String[] inputPaths = {
                "//src/data/dataset.csv",
                //"src/data/dataset_apenas_novos.csv"
        };
        String[] outputPaths = {"src/data/labels.csv"};

        model.train(inputPaths, outputPaths);
        //testar(model);
    }

    public static void testar(MLP23 model) {
        System.out.println("\n--- Modo de Teste Interativo ---");
        Scanner sc = new Scanner(System.in);
        int numTests = 0;

        while (true) {
            try {
                System.out.print("Digite o número de imagens que deseja testar: ");
                numTests = sc.nextInt();
                sc.nextLine();
                break;
            } catch (Exception e) {
                sc.nextLine();
            }
        }

        for (int i = 0; i < numTests; i++) {
            System.out.printf("\n--- Teste #%d ---\n", i + 1);
            System.out.println("Cole os 400 valores:");
            String line = sc.nextLine();
            String[] stringValues = line.split(",");

            if (stringValues.length != 400) {
                System.err.println("Erro: Tamanho incorreto.");
                continue;
            }

            double[][] testInput = new double[1][400];
            for (int j = 0; j < stringValues.length; j++) {
                testInput[0][j] = Double.parseDouble(stringValues[j].trim()) ;
            }

            Matrix prediction = model.getMLP().predict(new Matrix(testInput));
            double rawValue = prediction.get(0, 0);
            long predictedLabel = Math.round(rawValue);

            System.out.printf("Valor Bruto: %.4f\n", rawValue);
            if (predictedLabel == 0) System.out.println(">> Classificado como: 2");
            else System.out.println(">> Classificado como: 3");
        }
        sc.close();
    }
}