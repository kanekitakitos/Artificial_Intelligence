package neural.activation;

import java.util.function.Function;

/**
 * Implementa a função de ativação da Tangente Hiperbólica (TanH).
 * <p>
 * A TanH é frequentemente preferida em relação à Sigmoid para camadas ocultas, pois a sua saída
 * é centrada em zero (intervalo de -1 a 1). Isso pode ajudar a acelerar a convergência durante o treino,
 * evitando que os gradientes se desloquem consistentemente numa única direção.
 * </p>
 * <ul>
 *   <li><b>Função:</b> {@code f(z) = tanh(z)}</li>
 *   <li><b>Derivada:</b> {@code f'(y) = 1 - y^2}, onde {@code y = f(z)}</li>
 * </ul>
 * @author  Brandon Mejia
 * @version 2025-12-04
 */
public class TanH implements IDifferentiableFunction {
    @Override
    public Function<Double, Double> fnc() { return Math::tanh; }
    @Override
    public Function<Double, Double> derivative() { return y -> 1.0 - (y * y); }
}