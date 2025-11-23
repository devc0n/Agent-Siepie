package nl.devc0n.machinelearning.siepie.agent;

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import nl.devc0n.machinelearning.siepie.memory.ReplayBuffer;
import nl.devc0n.machinelearning.siepie.model.Action;
import nl.devc0n.machinelearning.siepie.model.Episode;
import nl.devc0n.machinelearning.siepie.model.GameStep;
import nl.devc0n.machinelearning.siepie.network.DQNNetwork;
import nl.devc0n.machinelearning.siepie.reward.RewardShaper;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.IOException;
import java.util.List;

@Slf4j
public class AgentSiepie {

    // Hyperparameters
    // Training config
    private static final int BATCH_SIZE = 64;  // Was dit misschien 32?
    private static final int TRAIN_FREQUENCY = 50;  // Niet te vaak
    private static final int TARGET_UPDATE_FREQUENCY = 1000;  // Terug naar origineel
    // Epsilon decay
    private static final double EPSILON_START = 0.30;
    private static final double EPSILON_END = 0.05;
    private static final int EPSILON_DECAY_EPISODES = 1000;
    private final DQNNetwork network;
    private final ReplayBuffer replayBuffer;
    private final RewardShaper rewardShaper;
    // Getters for monitoring
    @Getter
    private double epsilon;
    @Getter
    private int totalSteps;
    @Getter
    private int episodeCount;

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
                replayBuffer.getTotalSteps() >= BATCH_SIZE) {
            List<GameStep> batch = replayBuffer.sampleBatch(BATCH_SIZE);
            network.train(batch);
        }

        // Update target network periodically
        if (totalSteps % TARGET_UPDATE_FREQUENCY == 0) {
            network.updateTargetNetwork();
            System.out.println("Target network updated at step " + totalSteps);
        }
    }

    public void endEpisode(Episode episode, int finalScore, int episodeNum) {
        episode.finish(finalScore);


        // Mark last 5 frames as terminal states with death penalty
        rewardShaper.applyRewards(episode);

        // Add to replay buffer
        replayBuffer.addEpisode(episode);

        // Decay exploration rate
        epsilon = getEpsilon(episodeNum);

    }

    public void save(String path) throws IOException {
        network.save(path);
    }

    public void load(String path) throws IOException {
        network.load(path);
    }

    public int getBufferSize() {
        return replayBuffer.getTotalSteps();
    }

    // Override epsilon voor nieuwe exploration phase
    private double getEpsilon(int episode) {
        //checkpoint 450
        episode += 450;
        // Episodes 450-650: boost exploration
        if (episode >= 450 && episode < 650) {
            return 0.25;  // Hoge exploratie
        }
        // Episodes 650-1000: decay terug naar laag
        if (episode >= 650 && episode < 1000) {
            double progress = (episode - 650) / 350.0;
            return 0.25 - (progress * 0.20);  // 0.25 → 0.05
        }
        return 0.05;
    }
}