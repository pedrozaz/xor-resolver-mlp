import nn.NeuralNetwork;
import utils.MnistLoader;

import java.util.Collections;
import java.util.List;

public class Main {
    public static void main(String[] args) {
        try {
            String trainImg = "data/train-images-idx3-ubyte";
            String trainLbl = "data/train-labels-idx1-ubyte";

            List<MnistLoader.MnistData> trainingSet = MnistLoader.load(trainImg, trainLbl, 5000);

            NeuralNetwork nn = new NeuralNetwork(784, 64, 10);

            System.out.println("Initializing train with 5000 images...");

            int epochs = 5;

            for (int i = 0; i < epochs; i++) {
                Collections.shuffle(trainingSet);

                for (MnistLoader.MnistData item : trainingSet) {
                    nn.train(item.data, item.label);
                }
                System.out.println("Epoch " + (i + 1) + " complete.");
            }

            System.out.println("\n--- Inference Test ---");
            int correct = 0;
            int totalTest = 10;

            for (int i = 0; i < totalTest; i++) {
                MnistLoader.MnistData sample = trainingSet.get(i);
                List<Double> output = nn.feedforward(sample.data);

                int predicted = 0;
                double maxVal = -1;
                for (int j = 0; j < output.size(); j++) {
                    if (output.get(j) > maxVal) {
                        maxVal = output.get(j);
                        predicted = j;
                    }
                }

                System.out.println("Real: " + sample.labelDigit + " | Expected: " + predicted +
                        " (Credibility: " + String.format("%.2f", maxVal) + ")");
                if (predicted == sample.labelDigit) correct++;
            }
            System.out.println("Accuracy: " + (double) correct/totalTest);
        } catch (Exception e) {
            e.printStackTrace();
            System.err.println(e.getMessage());
        }
    }
}
