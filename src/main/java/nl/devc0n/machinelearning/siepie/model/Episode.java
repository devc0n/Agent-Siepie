package nl.devc0n.machinelearning.siepie.model;

import lombok.Getter;

import java.util.ArrayList;
import java.util.List;

@Getter
public class Episode {
    private final List<GameStep> steps;
    private int finalScore;
    private int episodeLength;

    public Episode() {
        this.steps = new ArrayList<>();
    }

    public void addStep(GameStep step) {
        steps.add(step);
    }

    public void finish(int finalScore) {
        this.finalScore = finalScore;

        var framesToDelete = 16;
        int minStepsRequired = 5;

        // Only delete if we have more than the minimum required
        if (steps.size() > minStepsRequired) {
            // Delete up to 16 frames, but keep at least 5
            int actualDeleteCount = Math.min(framesToDelete, steps.size() - minStepsRequired);
            steps.subList(steps.size() - actualDeleteCount, steps.size()).clear();
        }

        this.episodeLength = steps.size();
    }
}
