package nl.devc0n.machinelearning.siepie.ai;

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
}
