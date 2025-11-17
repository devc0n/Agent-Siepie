package nl.devc0n.machinelearning.siepie.network;

import nl.devc0n.machinelearning.siepie.model.Action;
import nl.devc0n.machinelearning.siepie.model.GameStep;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;


public class DQNNetwork {
    private MultiLayerNetwork model;
    private MultiLayerNetwork targetModel;

    private static final int INPUT_HEIGHT = 84;
    private static final int INPUT_WIDTH = 84;
    private static final int FRAME_STACK = 4;
    private static final double LEARNING_RATE = 0.00025;
    private static final double GAMMA = 0.99;

    public DQNNetwork() {
        buildNetwork();
    }

    private void buildNetwork() {
        int numActions = Action.getNumActions();

        var conf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(LEARNING_RATE))
                .list()
                .layer(new ConvolutionLayer.Builder(8, 8)
                        .nIn(FRAME_STACK)
                        .stride(4, 4)
                        .nOut(32)
                        .activation(Activation.RELU)
                        .build())
                .layer(new ConvolutionLayer.Builder(4, 4)
                        .stride(2, 2)
                        .nOut(64)
                        .activation(Activation.RELU)
                        .build())
                .layer(new ConvolutionLayer.Builder(3, 3)
                        .stride(1, 1)
                        .nOut(64)
                        .activation(Activation.RELU)
                        .build())
                .layer(new DenseLayer.Builder()
                        .nOut(512)
                        .activation(Activation.RELU)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .nOut(numActions)
                        .activation(Activation.IDENTITY)
                        .build())
                .setInputType(org.deeplearning4j.nn.conf.inputs.InputType.convolutional(
                        INPUT_HEIGHT, INPUT_WIDTH, FRAME_STACK))
                .build();

        model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(100));

        targetModel = model.clone();
    }

    public Action selectAction(INDArray frameStack, double epsilon) {
        if (Math.random() < epsilon) {
            // Bias early exploration: 50% do nothing, 50% move
            if (Math.random() < 0.5) {
                return Action.NOTHING;
            }
            return Action.fromIndex(1 + (int)(Math.random() * 4));
        }

        INDArray batchedInput = frameStack.reshape(1, FRAME_STACK, INPUT_HEIGHT, INPUT_WIDTH);
        INDArray qValues = model.output(batchedInput);

        int bestAction = Nd4j.argMax(qValues, 1).getInt(0);
        return Action.fromIndex(bestAction);
    }

    public void train(List<GameStep> batch) {
        if (batch.isEmpty()) return;

        // Collect all states and targets
        List<INDArray> statesList = new ArrayList<>();
        List<INDArray> targetsList = new ArrayList<>();

        for (GameStep step : batch) {
            // Add batch dimension for forward pass
            INDArray batchedFrame = step.getFrameStack().reshape(1, FRAME_STACK, INPUT_HEIGHT, INPUT_WIDTH);
            INDArray currentQ = model.output(batchedFrame);

            float targetQ;
            if (step.isTerminal()) {
                targetQ = step.getReward();
            } else {
                INDArray batchedNextFrame = step.getNextFrameStack().reshape(1, FRAME_STACK, INPUT_HEIGHT, INPUT_WIDTH);
                INDArray nextQ = targetModel.output(batchedNextFrame);
                float maxNextQ = nextQ.maxNumber().floatValue();
                targetQ = step.getReward() + (float)GAMMA * maxNextQ;
            }

            INDArray target = currentQ.dup();
            target.putScalar(new int[]{0, step.getAction().index}, targetQ); // Note: [0, action_index] for batch

            statesList.add(step.getFrameStack().reshape(1,4,84,84));
            targetsList.add(target.getRow(0)); // Remove batch dimension from target
        }

        // Stack into batches - this will create [batchSize, K, H, W]
        INDArray states = Nd4j.vstack(statesList.toArray(new INDArray[0]));
        INDArray targets = Nd4j.vstack(targetsList.toArray(new INDArray[0]));

        model.fit(states, targets);
    }

    public void updateTargetNetwork() {
        targetModel = model.clone();
    }

    public void save(String path) throws IOException {
        model.save(new File(path));
    }

    public void load(String path) throws IOException {
        model = MultiLayerNetwork.load(new File(path), true);
        targetModel = model.clone();
    }
}
