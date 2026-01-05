package utils;

import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class MnistLoader {

    public static class MnistData {
        public double[] data;
        public double[] label;
        public int labelDigit;

        public MnistData(double[] data, double[] label, int labelDigit) {
            this.data = data;
            this.label = label;
            this.labelDigit = labelDigit;
        }
    }

    public static List<MnistData> load(String imgPath, String labelPath, int limit) throws IOException {
        List<MnistData> dataset = new ArrayList<>();

        try (DataInputStream imgIn = new DataInputStream(new FileInputStream(imgPath));
             DataInputStream labelIn = new DataInputStream(new FileInputStream(labelPath))) {

            int imgMagic = imgIn.readInt();
            int numImages = imgIn.readInt();
            int rows = imgIn.readInt();
            int cols = imgIn.readInt();

            int labelMagic = labelIn.readInt();
            int numLabels = labelIn.readInt();

            int totalPixels = rows * cols;

            int quantityToLoad = (limit == -1) ? numImages : limit;

            System.out.println("Loading: " + quantityToLoad + " images of " + rows
            + "x" + cols + "...");

            for (int i = 0; i < quantityToLoad; i++) {
                int labelByte =  labelIn.readUnsignedByte();

                double[] labelArray = new double[10];
                labelArray[labelByte] = 1.0;

                double[] imgData = new double[totalPixels];
                for (int j = 0; j < totalPixels; j++) {
                    imgData[j] = imgIn.readUnsignedByte() / 255.0;
                }

                dataset.add(new MnistData(imgData, labelArray, labelByte));
            }
        }
        return dataset;
    }
}
