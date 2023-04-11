package com.ch.ml;

import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Collections;

public class SimpleMLModel {

    public static MultiLayerNetwork createXorModel() {
        // Create the network configuration
        NeuralNetConfiguration.ListBuilder configBuilder = new NeuralNetConfiguration.Builder()
                .seed(123)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Sgd(0.1))
                .list();

        // Add layers to the configuration
        configBuilder.layer(0, new DenseLayer.Builder()
                .nIn(2)
                .nOut(2)
                .weightInit(WeightInit.XAVIER)
                .activation(Activation.SIGMOID)
                .build());

        configBuilder.layer(1, new OutputLayer.Builder()
                .nIn(2)
                .nOut(1)
                .weightInit(WeightInit.XAVIER)
                .activation(Activation.SIGMOID)
                .lossFunction(LossFunctions.LossFunction.XENT)
                .build());

        // Build the network
        MultiLayerNetwork net = new MultiLayerNetwork(configBuilder.build());
        net.init();
        return net;
    }

//    public static DataSetIterator createXorDataSetIterator() {
//        // XOR input and output data
//        double[][] input = new double[][]{
//                {0, 0},
//                {0, 1},
//                {1, 0},
//                {1, 1}
//        };
//
//        double[][] output = new double[][]{
//                {0},
//                {1},
//                {1},
//                {0}
//        };
//
//        // Create the dataset iterator
//        return new ListDataSetIterator<>(Nd4j.createDataSetFromArrays(Nd4j.create(input), Nd4j.create(output)).asList(), 1);
//    }
public static DataSetIterator createXorDataSetIterator() {
    double[][] input = new double[][]{
            {0, 0},
            {0, 1},
            {1, 0},
            {1, 1}
    };

    double[][] output = new double[][]{
            {0},
            {1},
            {1},
            {0}
    };

    INDArray inputNDArray = Nd4j.create(input);
    INDArray outputNDArray = Nd4j.create(output);

    DataSet dataSet = new DataSet(inputNDArray, outputNDArray);

    // Normalize the input data
    DataNormalization normalizer = new NormalizerMinMaxScaler(0, 1);
    normalizer.fit(dataSet);
    normalizer.transform(dataSet);

    // Create a ListDataSetIterator with a batch size of 1
    return new ListDataSetIterator<>(Collections.singletonList(dataSet), 1);
}
    public static void trainModel(MultiLayerNetwork model, DataSetIterator iterator, int epochs) {
        for (int i = 0; i < epochs; i++) {
            iterator.reset();
            model.fit(iterator);
        }
    }

}
