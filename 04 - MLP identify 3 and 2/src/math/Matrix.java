package math;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Objects;
import java.util.Random;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.stream.IntStream;
/**
 * A pure Java implementation of a 2D matrix with parallelized mathematical operations.
 * <p>
 * This class provides a foundational set of matrix functionalities, including creation,
 * element-wise operations, dot products, and transformations. It is designed to be
 * a self-contained and efficient alternative to external libraries for basic to intermediate
 * linear algebra tasks.
 * </p>
 * <p>
 * A key feature of this implementation is the extensive use of Java's Parallel Streams
 * (`IntStream.range(0, rows).parallel()`) to accelerate computations on multi-core processors.
 * Operations like dot products, element-wise additions, and function applications are
 * automatically distributed across available cores, significantly improving performance
 * for large matrices.
 * </p>
 *
 * <h3>Example Usage</h3>
 * <p>
 * The following example demonstrates creating matrices and performing a basic dot product.
 * </p>
 * <pre>{@code
 * // Create a 2x3 matrix with random values
 * Matrix a = Matrix.Rand(2, 3, 123);
 *
 * // Create a 3x2 matrix from a 2D array
 * Matrix b = new Matrix(new double[][]{
 *     {1.0, 2.0},
 *     {3.0, 4.0},
 *     {5.0, 6.0}
 * });
 *
 * // Perform a dot product (a * b)
 * Matrix result = a.dot(b); // Result will be a 2x2 matrix
 *
 * System.out.println(result);
 * }</pre>
 *
 * @author hdaniel@ualg.pt, Brandon Mejia
 * @version 2025-12-05
 */
public class Matrix implements Serializable {

    private static final long serialVersionUID = 1L;
    private double[][] data;
    private int rows, cols;

    public Matrix(int rows, int cols) {
        data = new double[rows][cols];
        this.rows = rows;
        this.cols = cols;
    }


    public Matrix(double[][] data) {
        this.rows = data.length;
        // Ensure cols is defined even if there are no rows, and find the maximum number of columns.
        this.cols = (rows > 0) ? Arrays.stream(data).mapToInt(row -> row.length).max().orElse(0) : 0;
        this.data = new double[rows][cols];
        // Copy each row, respecting its individual length.
        // Shorter rows will be padded with zeros at the end.
        for (int i = 0; i < rows; i++) {
            System.arraycopy(data[i], 0, this.data[i], 0, data[i].length);
        }

    }


    static public Matrix Rand(int rows, int cols, int seed) {
        Matrix out = new Matrix(rows, cols);

        if (seed < 0)
            seed = (int) System.currentTimeMillis();

        return Rand(rows, cols, new Random(seed));
    }

    static public Matrix Rand(int rows, int cols, Random rand) {
        Matrix out = new Matrix(rows, cols);

        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                out.data[i][j] = rand.nextDouble();

        return out;

    }


    //accessors
    public double get(int row, int col) {
        return data[row][col];
    }
    public int rows() { return rows; }
    public int cols() { return cols; }


    //==============================================================
    //  Element operations
    //==============================================================

    //Apply Function<Double, Double> to all elements of the matrix
    //store the result in matrix result
    private Matrix traverse(Function<Double, Double> fnc) {
        Matrix result = new Matrix(rows, cols);

        IntStream.range(0, rows).parallel().forEach(i -> {
            Arrays.setAll(result.data[i], j -> fnc.apply(this.data[i][j]));
        });
        return result;
    }

    public Matrix apply(Function<Double, Double> fnc) {
        return traverse(fnc);
    }

    //multiply matrix by scalar
    public Matrix mult(double scalar) {
        return this.traverse(e -> e * scalar);
    }

    //add scalar to matrix
    public Matrix add(double scalar) {
        return this.traverse(e -> e + scalar);
    }

    //sub matrix from scalar:   scalar - M
    public Matrix subFromScalar(double scalar) {
        return this.traverse(e -> scalar - e);
    }


    //==============================================================
    //  Element-wise operations between two matrices
    //==============================================================

    //Element wise operation
    private Matrix elementWise(Matrix other, BiFunction<Double, Double, Double> fnc) {
        if (this.rows != other.rows || this.cols != other.cols) {
            throw new IllegalArgumentException("Incompatible matrix sizes for element wise.");
        }

        Matrix result = new Matrix(rows, cols);

        IntStream.range(0, rows).parallel().forEach(i -> {
            Arrays.setAll(result.data[i], j -> fnc.apply(this.data[i][j], other.data[i][j]));
        });
        return result;
    }

    //add two matrices
    public Matrix add(Matrix other) {
        return this.elementWise(other, (a, b) -> a + b);
    }

    //multiply two matrices (element wise)
    public Matrix mult(Matrix other) {
        return this.elementWise(other, (a, b) -> a * b);
    }

    //sub two matrices
    public Matrix sub(Matrix other) {
        return this.elementWise(other, (a, b) -> a - b);
    }


    //==============================================================
    //  Other math operations
    //==============================================================

    //sum all elements of the matrix
    public double sum() {
        // Parallelizes the sum of all matrix elements.
        return Arrays.stream(data).parallel().flatMapToDouble(Arrays::stream).sum();
    }

    //Sum by columns
    public Matrix sumColumns() {
        Matrix result = new Matrix(1, this.cols);

        IntStream.range(0, this.cols).parallel().forEach(j -> {
            double sum = 0;
            for(int i = 0; i < this.rows; i++) sum += this.data[i][j];
            result.data[0][j] = sum;
        });
        return result;
    }

    //Add row vector to each row of the matrix
    public Matrix addRowVector(Matrix rowVector) {
        if (rowVector.rows() != 1 || rowVector.cols() != this.cols) {
            throw new IllegalArgumentException("Incompatible sizes for adding row vector.");
        }
        Matrix result = new Matrix(this.rows, this.cols);

        IntStream.range(0, this.rows).parallel().forEach(i -> {
            Arrays.setAll(result.data[i], j -> this.data[i][j] + rowVector.data[0][j]);
        });
        return result;
    }


    //multiply two matrices (dot product)
    public Matrix dot(Matrix other) {
        if (this.cols != other.rows) {
            throw new IllegalArgumentException("Incompatible matrix sizes for multiplication.");
        }

        Matrix result = new Matrix(this.rows, other.cols);

        IntStream.range(0, this.rows).parallel().forEach(i -> {
            for (int j = 0; j < other.cols; j++) {
                double sum = 0;
                for (int k = 0; k < this.cols; k++) {
                    sum += this.data[i][k] * other.data[k][j];
                }
                result.data[i][j] = sum;
            }
        });
        return result;
    }


    //==============================================================
    //  Column and row operations
    //==============================================================

    //transpose matrix
    public Matrix transpose() {
        Matrix result = new Matrix(cols, rows);

        IntStream.range(0, rows).parallel().forEach(i -> {
            for (int j = 0; j < cols; j++) result.data[j][i] = data[i][j];
        });
        return result;
    }


    //==============================================================
    //  Convert operations
    //==============================================================

    public String toString() {
        StringBuilder sb = new StringBuilder();
        for (double[] row : data) {
            for (double val : row) {
                sb.append(String.format("%.3f ", val));
            }
            sb.append("\n");
        }
        return sb.toString();
    }
    
    @Override
    public Matrix clone() {
        return new Matrix(this.data);
    }


    //==============================================================
    //  Compare operations
    //==============================================================

    @Override
    public boolean equals(Object o) {
        if (!(o instanceof Matrix matrix)) return false;
        return rows == matrix.rows && cols == matrix.cols && Objects.deepEquals(data, matrix.data);
    }

    @Override
    public int hashCode() {
        return Objects.hash(Arrays.deepHashCode(data), rows, cols);
    }


    /**
     * Returns the underlying 2D double array of the matrix.
     * @return A 2D double array representing the matrix data.
     */
    public double[][] getData() {
        return this.data;
    }

}
