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

    public void train(double[] input_array, double[] target_array) {
        Matrix inputs = Matrix.fromArray(input_array);

        Matrix hidden = Matrix.mult(this.weights_ih, inputs);
        hidden.add(this.bias_h);
        hidden.map(this::sigmoid);

        Matrix outputs = Matrix.mult(this.weights_ho, hidden);
        outputs.add(this.bias_o);
        outputs.map(this::sigmoid);

        Matrix targets = Matrix.fromArray(target_array);
        Matrix output_errors = Matrix.sub(targets, outputs);

        Matrix gradients = Matrix.map(outputs, this::dsigmoid);
        gradients.mult(output_errors);
        gradients.mult(this.learning_rate);

        Matrix hidden_T = Matrix.trans(hidden);
        Matrix weight_ho_deltas = Matrix.mult(gradients, hidden_T);

        this.weights_ho.add(weight_ho_deltas);
        this.bias_o.add(gradients);

        Matrix who_T = Matrix.trans(this.weights_ho);
        Matrix hidden_errors = Matrix.mult(who_T, output_errors);

        Matrix hidden_gradient = Matrix.map(hidden, this::dsigmoid);
        hidden_gradient.mult(hidden_errors);
        hidden_gradient.mult(this.learning_rate);

        Matrix inputs_T = Matrix.trans(inputs);
        Matrix weight_ih_deltas = Matrix.mult(hidden_gradient, inputs_T);

        this.weights_ih.add(weight_ih_deltas);
        this.bias_h.add(hidden_gradient);
    }

    private double dsigmoid(double y) {
        return y * (1 - y);
    }

    private double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }
}
