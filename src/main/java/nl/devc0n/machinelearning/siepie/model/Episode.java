package nl.devc0n.machinelearning.siepie.model;



import java.util.ArrayList;
import java.util.List;

public class Episode {
    private List<GameStep> steps;
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

        // removes dead screen steps
        for (int i = 0; i < 16; i++) {
            int index = steps.size() -1;
            steps.remove(index);
        }

        this.episodeLength = steps.size();
    }

    // Getters
    public List<GameStep> getSteps() { return steps; }
    public int getFinalScore() { return finalScore; }
    public int getEpisodeLength() { return episodeLength; }
}
