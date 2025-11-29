package neural.activation;

import java.util.function.Function;


/**

 * @author Brandon Mejia
 * @version 29/11/2023
 */
public class ReLU implements IDifferentiableFunction {

    @Override
    public Function<Double, Double> fnc() {
        return x -> Math.max(0, x);
    }

    @Override
    public Function<Double, Double> derivative() {
        return x -> x > 0 ? 1.0 : 0.0;
    }
}