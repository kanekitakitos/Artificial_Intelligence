package neural.activation;

import java.util.function.Function;

/**
 * @author hdaniel@ualg.pt
 * @version 2025-11-05
 */
public interface IDifferentiableFunction {
    Function<Double, Double> fnc();
    Function<Double, Double> derivative();
}
