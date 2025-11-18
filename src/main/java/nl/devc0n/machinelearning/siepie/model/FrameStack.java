package nl.devc0n.machinelearning.siepie.model;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.awt.image.BufferedImage;
import java.util.ArrayDeque;
import java.util.Deque;

@Slf4j
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

//    public INDArray toINDArray() {
//        var start = System.currentTimeMillis();
//        BufferedImage[] frames = asArray();
//        int K = frames.length;
//        int H = frames[0].getHeight();
//        int W = frames[0].getWidth();
//        INDArray arr = Nd4j.create(K, H, W);
//
//        int idx = 0;
//        for (BufferedImage img : frames) {
//            // Get all pixels at once
//            int[] pixels = img.getRGB(0, 0, W, H, null, 0, W);
//
//            for (int y = 0; y < H; y++) {
//                for (int x = 0; x < W; x++) {
//                    int rgb = pixels[y * W + x];
//                    int gray = rgb & 0xFF;
//                    arr.putScalar(new int[]{idx, y, x}, gray / 255.0);
//                }
//            }
//            idx++;
//        }
//        log.debug("Frame stack took {}ms", System.currentTimeMillis() - start);
//        return arr;
//    }

    public INDArray toINDArray() {
        BufferedImage[] frames = asArray();
        int K = frames.length;                 // number of stacked frames (4)
        int H = frames[0].getHeight();         // 128
        int W = frames[0].getWidth();          // 128
        int C = 3;                             // RGB channels

        // Final shape: [K*C, H, W] = [12, 128, 128]
        // This matches what the network expects: INPUT_CHANNELS=12
        INDArray arr = Nd4j.create(K * C, H, W);

        for (int f = 0; f < K; f++) {
            BufferedImage img = frames[f];

            // Bulk extraction of all pixels
            int[] pixels = img.getRGB(0, 0, W, H, null, 0, W);

            for (int i = 0; i < pixels.length; i++) {
                int pixel = pixels[i];

                // Pixel location in HxW
                int y = i / W;
                int x = i % W;

                // Extract R G B (0-255) and normalize
                float r = ((pixel >> 16) & 0xFF) / 255.0f;
                float g = ((pixel >> 8) & 0xFF) / 255.0f;
                float b = (pixel & 0xFF) / 255.0f;

                // Put into flattened channel dimension
                // Frame 0: channels 0-2, Frame 1: channels 3-5, etc.
                arr.putScalar(new int[]{f * C + 0, y, x}, r);
                arr.putScalar(new int[]{f * C + 1, y, x}, g);
                arr.putScalar(new int[]{f * C + 2, y, x}, b);
            }
        }

        return arr;
    }


}
