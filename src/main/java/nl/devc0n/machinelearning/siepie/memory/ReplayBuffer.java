package nl.devc0n.machinelearning.siepie.memory;

import nl.devc0n.machinelearning.siepie.model.Episode;
import nl.devc0n.machinelearning.siepie.model.GameStep;

import java.util.*;

public class ReplayBuffer {

    private final Deque<Episode> episodes = new ArrayDeque<>();
    private final int maxEpisodes;
    private final Random random = new Random();

    public ReplayBuffer(int maxEpisodes) {
        this.maxEpisodes = maxEpisodes;
    }

    public ReplayBuffer() {
        this(1000); // Default: keep last 100 episodes
    }

    public void addEpisode(Episode episode) {
        episodes.addLast(episode);
        while (episodes.size() > maxEpisodes) {
            episodes.removeFirst(); // Remove oldest
        }
    }

    /**
     * Sample a batch of steps from stored episodes
     * Prioritizes sampling from better episodes (survived longer)
     */
    public List<GameStep> sampleBatch(int batchSize) {
        List<GameStep> batch = new ArrayList<>();

        // Prioritize sampling from better episodes
        List<Episode> weightedEpisodes = new ArrayList<>();
        for (Episode ep : episodes) {
            int weight = Math.max(1, ep.getEpisodeLength() / 50);
            for (int i = 0; i < weight; i++) {
                weightedEpisodes.add(ep);
            }
        }

        if (weightedEpisodes.isEmpty()) {
            return batch;
        }

        for (int i = 0; i < batchSize; i++) {
            Episode ep = weightedEpisodes.get(random.nextInt(weightedEpisodes.size()));
            if (!ep.getSteps().isEmpty()) {
                GameStep step = ep.getSteps().get(random.nextInt(ep.getSteps().size()));
                batch.add(step);
            }
        }

        return batch;
    }

    public int getTotalSteps() {
        return episodes.stream().mapToInt(e -> e.getSteps().size()).sum();
    }

    public int getEpisodeCount() {
        return episodes.size();
    }

    public void clear() {
        episodes.clear();
    }
}
