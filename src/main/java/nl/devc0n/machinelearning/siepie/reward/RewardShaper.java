package nl.devc0n.machinelearning.siepie.reward;

import nl.devc0n.machinelearning.siepie.model.Episode;
import nl.devc0n.machinelearning.siepie.model.GameStep;

public class RewardShaper {
    private static final float SURVIVAL_BONUS = 0.01f;
    private static final float DEATH_PENALTY = -10.0f;
    private static final int TERMINAL_WINDOW = 5;

    public float calculateStepReward(boolean isTerminal) {
        if (isTerminal) {
            return DEATH_PENALTY;
        }
        return SURVIVAL_BONUS;
    }

    /**
     * After death detected, mark the last N frames as terminal states
     * This handles the delayed death detection problem
     */
    public void markTerminalStates(Episode episode) {
        if (episode.getSteps().isEmpty()) return;

        int episodeLength = episode.getSteps().size();
        int terminalStart = Math.max(0, episodeLength - TERMINAL_WINDOW);

        // Mark last N steps as terminal and apply death penalty
        for (int i = terminalStart; i < episodeLength; i++) {
            GameStep step = episode.getSteps().get(i);
            step.setTerminal(true);
            step.setReward(DEATH_PENALTY);
        }
    }
}
