package nl.devc0n.machinelearning.siepie.agent;

import nl.devc0n.machinelearning.siepie.memory.ReplayBuffer;
import nl.devc0n.machinelearning.siepie.model.Action;
import nl.devc0n.machinelearning.siepie.model.Episode;
import nl.devc0n.machinelearning.siepie.model.GameStep;
import nl.devc0n.machinelearning.siepie.network.DQNNetwork;
import nl.devc0n.machinelearning.siepie.reward.RewardShaper;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.IOException;
import java.util.List;

public class AgentSiepie {
    private DQNNetwork network;
    private ReplayBuffer replayBuffer;
    private RewardShaper rewardShaper;

    private double epsilon;
    private int totalSteps;
    private int episodeCount;

    // Hyperparameters
    private static final double EPSILON_START = 1.0;
    private static final double EPSILON_DECAY = 0.995;
    private static final double EPSILON_MIN = 0.1;
    private static final int BATCH_SIZE = 64;
    private static final int TARGET_UPDATE_FREQUENCY = 2000;
    private static final int TRAIN_FREQUENCY = 100;
    private static final int MIN_BUFFER_SIZE = 500;

    public AgentSiepie(ReplayBuffer replayBuffer) {
        this.network = new DQNNetwork();
        this.rewardShaper = new RewardShaper();
        this.epsilon = EPSILON_START;
        this.totalSteps = 0;
        this.episodeCount = 0;
        this.replayBuffer = replayBuffer;
    }

    public void startEpisode() {
        episodeCount++;
    }

    public Action selectAction(INDArray frameStack) {
        return network.selectAction(frameStack, epsilon);
    }

    public void recordStep(Episode currentEpisode, INDArray frameStack,
                           Action action, INDArray nextFrameStack) {
        GameStep step = new GameStep(frameStack, action, nextFrameStack,
                currentEpisode.getSteps().size());
        currentEpisode.addStep(step);

        totalSteps++;

        // Train periodically
        if (totalSteps % TRAIN_FREQUENCY == 0 &&
                replayBuffer.getTotalSteps() >= MIN_BUFFER_SIZE) {
            List<GameStep> batch = replayBuffer.sampleBatch(BATCH_SIZE);
            network.train(batch);
        }

        // Update target network periodically
        if (totalSteps % TARGET_UPDATE_FREQUENCY == 0) {
            network.updateTargetNetwork();
            System.out.println("Target network updated at step " + totalSteps);
        }
    }

    public void endEpisode(Episode episode, int finalScore) {
        episode.finish(finalScore);

        // Mark last 5 frames as terminal states with death penalty
        rewardShaper.markTerminalStates(episode);

        // Add to replay buffer
        replayBuffer.addEpisode(episode);

        // Decay exploration rate
        epsilon = Math.max(EPSILON_MIN, epsilon * EPSILON_DECAY);

        // Log progress
        System.out.printf(
                "%-10d %-10d %-10d %-10.3f %-15d%n",
                episodeCount,
                finalScore,
                episode.getSteps().size(),
                epsilon,
                replayBuffer.getEpisodeCount()
        );

    }

    public void save(String path) throws IOException {
        network.save(path);
    }

    public void load(String path) throws IOException {
        network.load(path);
    }

    // Getters for monitoring
    public double getEpsilon() { return epsilon; }
    public int getTotalSteps() { return totalSteps; }
    public int getEpisodeCount() { return episodeCount; }
    public int getBufferSize() { return replayBuffer.getTotalSteps(); }
}