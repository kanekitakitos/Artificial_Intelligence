package neural.activation;

import java.util.function.Function;

public class HyperbolicTangent implements IDifferentiableFunction {


    public double fnc(double v) {
        return Math.tanh(v);
    }


    public double derivative(double v) {
        // A derivada de tanh(x) é 1 - tanh(x)^2
        double t = Math.tanh(v);
        return 1.0 - (t * t);
    }

    @Override
    public Function<Double, Double> fnc() {
        return this::fnc;
    }

    @Override
    public Function<Double, Double> derivative() {
        return this::derivative;
    }
}