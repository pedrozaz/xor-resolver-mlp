package nn;

import math.Matrix;

import java.util.ArrayList;
import java.util.List;

public class NeuralNetwork {
    private int input_nodes;
    private int hidden_nodes;
    private int output_nodes;

    private Matrix weights_ih;
    private Matrix weights_ho;

    private Matrix bias_h;
    private Matrix bias_o;

    private double learning_rate;

    public NeuralNetwork(int input_nodes, int hidden_nodes, int output_nodes) {
        this.input_nodes = input_nodes;
        this.hidden_nodes = hidden_nodes;
        this.output_nodes = output_nodes;
        this.learning_rate = 0.1;

        this.weights_ih = new Matrix(this.hidden_nodes, this.input_nodes);
        this.weights_ho = new Matrix(this.output_nodes, this.hidden_nodes);

        this.weights_ih.randomize();
        this.weights_ho.randomize();

        this.bias_h = new Matrix(this.hidden_nodes, 1);
        this.bias_o = new Matrix(this.output_nodes, 1);

        this.bias_h.randomize();
        this.bias_o.randomize();
    }

    public List<Double> feedforward(double[] input_array) {
        Matrix inputs = Matrix.fromArray(input_array);

        Matrix hidden = Matrix.mult(this.weights_ih, inputs);
        hidden.add(this.bias_h);
        hidden.map(this::sigmoid);

        Matrix output =  Matrix.mult(this.weights_ho, hidden);
        output.add(this.bias_o);
        output.map(this::sigmoid);

        ArrayList<Double> outputList = new ArrayList<>();
        for (int i = 0; i < output.rows; i++) {
            outputList.add(output.data[i][0]);
        }

        return outputList;
    }

    private double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }
}
