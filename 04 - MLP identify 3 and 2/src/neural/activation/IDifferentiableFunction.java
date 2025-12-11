package neural.activation;

import java.io.Serializable;
import java.util.function.Function;

/**
 * @author hdaniel@ualg.pt
 * @version 202511051244
 */
public interface IDifferentiableFunction extends Serializable {
    Function<Double, Double> fnc();
    Function<Double, Double> derivative();
}
