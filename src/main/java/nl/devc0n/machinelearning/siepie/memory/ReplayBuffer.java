package nl.devc0n.machinelearning.siepie.memory;

import nl.devc0n.machinelearning.siepie.model.Episode;
import nl.devc0n.machinelearning.siepie.model.GameStep;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class ReplayBuffer {
    private List<Episode> episodes;
    private int maxEpisodes;
    private Random random;

    public ReplayBuffer(int maxEpisodes) {
        this.episodes = new ArrayList<>();
        this.maxEpisodes = maxEpisodes;
        this.random = new Random();
    }

    public ReplayBuffer() {
        this(100); // Default: keep last 100 episodes
    }

    public void addEpisode(Episode episode) {
        episodes.add(episode);

        // Remove oldest episodes if buffer too large
        while (episodes.size() > maxEpisodes) {
            episodes.remove(0);
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
