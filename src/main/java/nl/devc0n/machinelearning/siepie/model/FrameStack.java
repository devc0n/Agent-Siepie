package nl.devc0n.machinelearning.siepie.model;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
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

    public INDArray toINDArray() {
        var start = System.currentTimeMillis();
        BufferedImage[] frames = asArray();
        int K = frames.length;
        int H = frames[0].getHeight();
        int W = frames[0].getWidth();
        INDArray arr = Nd4j.create(K, H, W);

        int idx = 0;
        for (BufferedImage img : frames) {
            // For TYPE_BYTE_GRAY, raster is slightly faster than getRGB()
            byte[] pixels = ((DataBufferByte) img.getRaster().getDataBuffer()).getData();

            for (int y = 0; y < H; y++) {
                for (int x = 0; x < W; x++) {
                    // byte is signed (-128 to 127), convert to unsigned (0 to 255)
                    int gray = pixels[y * W + x] & 0xFF;
                    arr.putScalar(new int[]{idx, y, x}, gray / 255.0);
                }
            }
            idx++;
        }
        log.debug("Frame stack took {}ms", System.currentTimeMillis() - start);
        return arr;
    }
}
