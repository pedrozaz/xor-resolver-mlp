import nn.NeuralNetwork;

import java.util.List;
import java.util.Random;

public class Main {
    public static void main(String[] args) {
        NeuralNetwork nn = new NeuralNetwork(2, 3, 1);

        double[][] X = {
                {0, 0},
                {0, 1},
                {1, 0},
                {1, 1}
        };

        double[][] Y = {
                {0},
                {1},
                {1},
                {0}
        };

        System.out.println("Training...");

        int epochs = 50000;
        Random random = new Random();

        for (int i = 0; i < epochs; i++) {
            int index = random.nextInt(4);

            nn.train(X[index], Y[index]);

            if ((i % 10000) == 0) {
                System.out.println("Epoch " + i);
            }
        }

        System.out.println("Training done.");

        System.out.println("XOR Results:");
        for (double[] input : X) {
            List<Double> output = nn.feedforward(input);

            String i1 = String.format("%.0f", input[0]);
            String i2 = String.format("%.0f", input[1]);
            double prediction = output.get(0);

            System.out.printf("[%s, %s] -> %.4f (Expected: %.0f)%n",
                    i1, i2, prediction, (prediction > 0.5 ? 1.0 : 0.0));
        }
    }
}
