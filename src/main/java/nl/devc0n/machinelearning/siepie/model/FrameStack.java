package nl.devc0n.machinelearning.siepie.model;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.awt.image.BufferedImage;
import java.util.ArrayDeque;
import java.util.Deque;

public class FrameStack {
    /**
     * Queue of buffered images. The queue is ordered from oldest to newest image.
     */
    private final Deque<BufferedImage> q;

    /**
     * Maximum size of the queue.
     */
    private final int k;

    public FrameStack(int k, BufferedImage initial) {
        this.k = k;
        q = new ArrayDeque<>(k);
        for (int i = 0; i < k; i++) q.addLast(initial);
    }

    public void push(BufferedImage img) {
        if (q.size() == k) q.removeFirst();
        q.addLast(img);
    }

    public BufferedImage[] asArray() {
        return q.toArray(new BufferedImage[0]);
    }

    public INDArray toINDArray() {
        BufferedImage[] frames = asArray();
        int K = frames.length;
        int H = frames[0].getHeight();
        int W = frames[0].getWidth();
        INDArray arr = Nd4j.create(K, H, W);

        int idx = 0;
        for (BufferedImage img : frames) {
            // Get all pixels at once
            int[] pixels = img.getRGB(0, 0, W, H, null, 0, W);

            for (int y = 0; y < H; y++) {
                for (int x = 0; x < W; x++) {
                    int rgb = pixels[y * W + x];
                    int gray = rgb & 0xFF;
                    arr.putScalar(new int[]{idx, y, x}, gray / 255.0);
                }
            }
            idx++;
        }
        return arr;
    }
}
