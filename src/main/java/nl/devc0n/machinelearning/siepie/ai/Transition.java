package nl.devc0n.machinelearning.siepie.ai;

import org.nd4j.linalg.api.ndarray.INDArray;

public record Transition(INDArray state, int action, double reward, INDArray nextState, boolean done) {}
