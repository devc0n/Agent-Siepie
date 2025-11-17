package nl.devc0n.machinelearning.siepie.model;



import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

public class Episode implements Comparable<Episode> {
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

        var framesToDelete = 13;
        if (steps.size() < framesToDelete){
            if (steps.size() > 5){
                framesToDelete = steps.size() - 5;
            }else{
                framesToDelete = steps.size();
            }
        }
        // removes dead screen steps
        for (int i = 0; i < framesToDelete; i++) {
            int index = steps.size() -1;
            steps.remove(index);
        }
        this.episodeLength = steps.size();
    }

    // Getters
    public List<GameStep> getSteps() { return steps; }
    public int getFinalScore() { return finalScore; }
    public int getEpisodeLength() { return episodeLength; }

    @Override
    public int compareTo(Episode o) {
        return Integer.compare(this.finalScore, o.finalScore);
    }
}
