package nl.devc0n.machinelearning.siepie.memory;

import lombok.Getter;

import java.util.LinkedList;
import java.util.Queue;

/**
 * Tracks moving average of scores for ML training logs
 */
public class MovingAverageTracker {
    private final Queue<Integer> scoreWindow;
    private final int windowSize;
    private int sum;
    @Getter
    private int hiScore;

    public MovingAverageTracker(int windowSize) {
        this.windowSize = windowSize;
        this.scoreWindow = new LinkedList<>();
        this.sum = 0;
    }

    /**
     * Add a new score and return the current moving average
     *
     * @param score The episode score
     * @return The moving average over the window
     */
    public double addScore(int score) {
        scoreWindow.add(score);
        sum += score;

        // Remove the oldest score if window is full
        if (scoreWindow.size() > windowSize) {
            sum -= scoreWindow.poll();
        }

        if (score > hiScore) hiScore = score;

        return getAverage();
    }

    /**
     * Get current moving average
     */
    public double getAverage() {
        if (scoreWindow.isEmpty()) {
            return 0.0;
        }
        return (double) sum / scoreWindow.size();
    }

    /**
     * Get current window size (useful for first N episodes)
     */
    public int getCurrentWindowSize() {
        return scoreWindow.size();
    }

    /**
     * Reset the tracker
     */
    public void reset() {
        scoreWindow.clear();
        sum = 0;
    }
}
