package math;

import java.util.function.Function;

public class Matrix {
    public int rows;
    public int cols;
    public double[][] data;

    public Matrix(int rows, int cols) {
        this.rows = rows;
        this.cols = cols;
        this.data = new double[rows][cols];
    }

    public void randomize() {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                this.data[i][j] = Math.random() * 2 - 1;
            }
        }
    }

    public void add(Matrix o) {
        if (this.rows != o.rows | this.cols != o.cols) {
            throw new IllegalArgumentException("Incompatible dimensions.");
        }
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                this.data[i][j] += o.data[i][j];
            }
        }
    }

    public static Matrix sub(Matrix a, Matrix b) {
        Matrix y = new Matrix(a.rows, b.cols);
        for (int i = 0; i < a.rows; i++) {
            for (int j = 0; j < b.cols; j++) {
                y.data[i][j] = a.data[i][j] - b.data[i][j];
            }
        }
        return y;
    }

    public static Matrix trans(Matrix a) {
        Matrix y = new Matrix(a.cols, a.rows);
        for (int i = 0; i < a.rows; i++) {
            for (int j = 0; j < a.cols; j++) {
                y.data[j][i] = a.data[i][j];
            }
        }
        return y;
    }

    public static Matrix mult(Matrix a, Matrix b) {
        if (a.cols != b.rows) {
            throw new IllegalArgumentException("Incompatible dimensions.");
        }
        Matrix y = new Matrix(a.rows, b.cols);
        for (int i = 0; i < a.rows; i++) {
            for (int j = 0; j < b.cols; j++) {
                double sum = 0;
                for (int k = 0; k < a.cols; k++) {
                    sum += a.data[i][k] * b.data[k][j];
                }
                y.data[i][j] = sum;
            }
        }
        return y;
    }

    public void mult(double s) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] *= s;
            }
        }
    }

    public void mult(Matrix o) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] *= o.data[i][j];
            }
        }
    }

    public void map(Function<Double, Double> func) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                this.data[i][j] = func.apply(data[i][j]);
            }
        }
    }

    public static Matrix map(Matrix m, Function<Double, Double> func) {
        Matrix y = new Matrix(m.rows, m.cols);
        for (int i = 0; i < y.rows; i++) {
            for (int j = 0; j < y.cols; j++) {
                y.data[i][j] = func.apply(m.data[i][j]);
            }
        }
        return y;
    }

    public static Matrix fromArray(double[] arr) {
        Matrix m = new Matrix(arr.length, 1);
        for (int i = 0; i < arr.length; i++) {
            m.data[i][0] = arr[i];
        }
        return m;
    }
}
