package nl.devc0n.machinelearning.siepie.ai;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.learning.config.Adam;

public class ModelFactory {
    public MultiLayerNetwork createPolicyNetwork(int inputChannels, int inputH, int inputW, int nActions) {
        var conf = new NeuralNetConfiguration.Builder()
                .updater(new Adam(1e-4))
                .seed(123)
                .list()
                .layer(new ConvolutionLayer.Builder(8, 8).stride(4, 4).nIn(inputChannels).nOut(32)
                        .activation(org.nd4j.linalg.activations.Activation.RELU).build())
                .layer(new ConvolutionLayer.Builder(4, 4).stride(2, 2).nOut(64)
                        .activation(org.nd4j.linalg.activations.Activation.RELU).build())
                .layer(new ConvolutionLayer.Builder(3, 3).stride(1, 1).nOut(64)
                        .activation(org.nd4j.linalg.activations.Activation.RELU).build())
                .layer(new DenseLayer.Builder().nOut(512).activation(org.nd4j.linalg.activations.Activation.RELU).build())
                .layer(new OutputLayer.Builder(org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction.MCXENT)
                        .activation(org.nd4j.linalg.activations.Activation.SOFTMAX).nOut(nActions).build())
                .setInputType(InputType.convolutional(inputH, inputW, inputChannels))
                .build();

        var net = new MultiLayerNetwork(conf);
        net.init();
        return net;
    }
}
