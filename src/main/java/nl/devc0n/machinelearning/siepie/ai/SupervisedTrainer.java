package nl.devc0n.machinelearning.siepie.ai;

import org.deeplearning4j.datasets.iterator.IteratorDataSetIterator;
import org.deeplearning4j.datasets.iterator.utilty.ListDataSetIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;

public class SupervisedTrainer {
    /** Simple conversion: frames: BufferedImage[K] -> INDArray shape [1, channels, H, W] */
    public static INDArray imageStackToINDArray(BufferedImage[] frames) {
        int K = frames.length;
        int H = frames[0].getHeight();
        int W = frames[0].getWidth();
        INDArray arr = Nd4j.create(1, K, H, W); // NCHW
        for (int k = 0; k < K; k++) {
            BufferedImage img = frames[k];
            for (int y = 0; y < H; y++) {
                for (int x = 0; x < W; x++) {
                    int rgb = img.getRGB(x,y);
                    int gray = rgb & 0xFF; // if grayscale
                    arr.putScalar(new int[]{0,k,y,x}, gray / 255.0);
                }
            }
        }
        return arr;
    }

    public static void train(MultiLayerNetwork net, List<BufferedImage[]> inputs, List<Integer> labels, int epochs) {
        List<DataSet> ds = new ArrayList<>();
        for (int i = 0; i < inputs.size(); i++) {
            INDArray in = imageStackToINDArray(inputs.get(i));
            INDArray out = Nd4j.zeros(1, 5); // 5 actions
            out.putScalar(0, labels.get(i), 1.0);
            ds.add(new DataSet(in, out));
        }
        var iter = new ListDataSetIterator(ds, 32) {
        };
        for (int e = 0; e < epochs; e++) {
            iter.reset();
            net.fit(iter);
            System.out.printf("Epoch %d done%n", e);
        }
    }
}
