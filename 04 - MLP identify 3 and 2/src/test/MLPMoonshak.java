package test;

import math.Matrix;
import neural.MLP;

import neural.activation.IDifferentiableFunction;
import neural.activation.Sigmoid;
import neural.activation.Step;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class MLPMoonshak {

    private MLP mlp;
    private Matrix trX;
    private Matrix trY;

    @BeforeEach
    void setUp() {
        // Configuração base da classe MLPNXOR para o problema NXOR
        double lr = 0.4;
        int epochs = 10000;
        int[] topology = {2, 2, 1};
        int seed = 4; // Semente para garantir resultados reproduzíveis nos testes

        // Dataset NXOR
        trX = new Matrix(
                new double[][]{
                        {0, 0},
                        {0, 1},
                        {1, 0},
                        {1, 1}});

        trY = new Matrix(
                new double[][]{
                        {1},
                        {0},
                        {0},
                        {1}});

        // Criar e treinar a MLP
        mlp = new MLP(topology,
                new IDifferentiableFunction[]{
                        new Sigmoid(),
                        new Sigmoid(),},
                seed);
        //mlp.train(trX, trY, lr, epochs);
    }


    @Test
    @DisplayName("Input (1, 1) should result in 1")
    void test01() {
        Matrix input = new Matrix(new double[][]{{1, 1}}); // Entrada para (1,1)
        Matrix prediction = mlp.predict(input);
        // Converte a probabilidade para uma classificação (0 ou 1)
        Matrix finalOutput = prediction.apply(new Step().fnc());
        Matrix expected = new Matrix(new double[][]{{1}});
        System.out.println(finalOutput.toString());
        assertEquals(expected, finalOutput);
    }

    @Test
    @DisplayName("Should handle batch predictions correctly")
    void test02() {
        Matrix testInputs = new Matrix(new double[][]{
                {0, 1},
                {1, 1},
                {0, 1}
        });
        Matrix expectedOutputs = new Matrix(new double[][]{
                {0},
                {1},
                {0}
        });
        Matrix prediction = mlp.predict(testInputs);
        // Converte as probabilidades para classificações (0 ou 1)
        Matrix finalOutput = prediction.apply(new Step().fnc());
        System.out.println(finalOutput.toString());
        assertEquals(expectedOutputs, finalOutput);
    }

    @Test
    @DisplayName("Should handle a larger batch prediction correctly")
    void test03() {
        Matrix testInputs = new Matrix(new double[][]{
                {1, 1},
                {0, 1},
                {1, 0},
                {1, 0},
                {0, 0},
                {1, 0}
        });
        Matrix expectedOutputs = new Matrix(new double[][]{
                {1}, {0}, {0}, {0}, {1}, {0}
        });
        Matrix prediction = mlp.predict(testInputs);
        // Converte as probabilidades para classificações (0 ou 1)
        Matrix finalOutput = prediction.apply(new Step().fnc());
        System.out.println(finalOutput.toString());
        assertEquals(expectedOutputs, finalOutput);
    }

    @Test
    @DisplayName("Should handle batch predictions with floating point inputs")
    void test04() {
        Matrix testInputs = new Matrix(new double[][]{
                {0.1, 0.9},
                {0.21, 0.01},
                {0.95, 0.89},
                {0.12, 0.85}
        });
        Matrix expectedOutputs = new Matrix(new double[][]{
                {0},
                {1},
                {1},
                {0}
        });
        Matrix prediction = mlp.predict(testInputs);
        // Converte as probabilidades para classificações (0 ou 1)
        Matrix finalOutput = prediction.apply(new Step().fnc());
        System.out.println(finalOutput.toString());
        assertEquals(expectedOutputs, finalOutput);
    }
}
