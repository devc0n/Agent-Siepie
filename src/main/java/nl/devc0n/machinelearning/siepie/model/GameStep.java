package nl.devc0n.machinelearning.siepie.model;

import org.nd4j.linalg.api.ndarray.INDArray;

public class GameStep {
    private INDArray frameStack;
    private Action action;
    private float reward;
    private INDArray nextFrameStack;
    private boolean terminal;
    private int stepNumber;

    public GameStep(INDArray frameStack, Action action, INDArray nextFrameStack, int stepNumber) {
        this.frameStack = frameStack;
        this.action = action;
        this.nextFrameStack = nextFrameStack;
        this.stepNumber = stepNumber;
        this.terminal = false;
        this.reward = 0.01f; // Default survival bonus
    }

    // Getters and setters
    public INDArray getFrameStack() { return frameStack; }
    public Action getAction() { return action; }
    public float getReward() { return reward; }
    public void setReward(float reward) { this.reward = reward; }
    public INDArray getNextFrameStack() { return nextFrameStack; }
    public boolean isTerminal() { return terminal; }
    public void setTerminal(boolean terminal) { this.terminal = terminal; }
    public int getStepNumber() { return stepNumber; }
}
