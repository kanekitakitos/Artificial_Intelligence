package math;

import java.util.Arrays;
import java.util.Objects;
import java.io.Serializable;
import java.util.Random;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.stream.IntStream;

/**
 * @author hdaniel@ualg.pt, Brandon Mejia
 * @version 202511052002
 */
public class Matrix implements Serializable {

    private double[][] data;
    private int rows, cols;

    public Matrix(int rows, int cols) {
        data = new double[rows][cols];
        this.rows = rows;
        this.cols = cols;
    }


    public Matrix(double[][] data) {
        this.rows = data.length;
        this.cols = data[0].length;
        this.data = new double[rows][cols];
        for (int i = 0; i < rows; i++)
            System.arraycopy(data[i], 0, this.data[i], 0, cols);

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

    /**
     * Returns a copy of the specified row as a double array.
     * @param row The index of the row to retrieve.
     * @return A new double array containing the data from the specified row.
     * @throws IllegalArgumentException if the row index is out of bounds.
     */
    public double[] get(int row) {
        if (row < 0 || row >= this.rows) {
            throw new IllegalArgumentException("Invalid row index: " + row);
        }
        return Arrays.copyOf(data[row], this.cols);
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

    //divide matrix by scalar
    public Matrix div(double scalar) {
        if (scalar == 0) {
            throw new IllegalArgumentException("Division by zero.");
        }
        return this.traverse(e -> e / scalar);
    }

    public Matrix pow(double exponent) {
        return this.traverse(e -> Math.pow(e, exponent));
    }

    public Matrix sqrt() {
        return this.traverse(Math::sqrt);
    }

    /**
     * Applies the absolute value function to each element of the matrix.
     * @return A new matrix where each element is the absolute value of the corresponding element in the original matrix.
     */
    public Matrix abs() {
        return this.traverse(Math::abs);
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

    //divide two matrices (element wise)
    public Matrix div(Matrix other) {
        return this.elementWise(other, (a, b) -> a / b);
    }


    //==============================================================
    //  Other math operations
    //==============================================================

    //sum all elements of the matrix
    public double sum() {
        // Paraleliza a soma de todos os elementos da matriz
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

    /**
     * Extracts a range of rows from the matrix to create a new sub-matrix.
     * This is equivalent to slicing the matrix along its rows.
     *
     * @param start The starting row index (inclusive).
     * @param end   The ending row index (exclusive).
     * @return A new {@link Matrix} containing the specified rows.
     */
    public Matrix getRows(int start, int end) {
        if (start < 0 || end > this.rows || start > end) {
            throw new IllegalArgumentException("Invalid row range for getRows.");
        }
        int numRows = end - start;
        double[][] subData = new double[numRows][this.cols];
        System.arraycopy(this.data, start, subData, 0, numRows);
        return new Matrix(subData);
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
