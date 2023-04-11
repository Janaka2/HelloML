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
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Arrays;
import java.util.List;

public class SimpleFeedForwardNN {

    public static void main(String[] args) {

        int numInput = 2; // Number of input features
        int numHiddenNodes = 10;
        int numOutputs = 1;

        MultiLayerNetwork model = new MultiLayerNetwork(new NeuralNetConfiguration.Builder()
                .seed(12345)
                .weightInit(WeightInit.XAVIER)
                .activation(Activation.RELU)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Sgd(0.1))
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(numInput)
                        .nOut(numHiddenNodes)
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .activation(Activation.IDENTITY)
                        .nIn(numHiddenNodes)
                        .nOut(numOutputs)
                        .build())
                .build());

        model.init();

        // Toy dataset for training
        INDArray input = Nd4j.create(new double[][]{
                {1, 1},
                {1, 2},
                {2, 1},
                {2, 2}});
        INDArray output = Nd4j.create(new double[][]{
                {2},
                {3},
                {3},
                {4}});

        DataSet trainingData = new DataSet(input, output);
        ListDataSetIterator<DataSet> iterator = new ListDataSetIterator<>(Arrays.asList(trainingData), 1);

        // Train the model
        for (int i = 0; i < 1000; i++) {
            model.fit(iterator);
            iterator.reset();
        }

        // New input data for prediction
        INDArray newInput = Nd4j.create(new double[][]{
                {3, 3},
                {4, 5}});

        // Make predictions
        INDArray predictions = model.output(newInput);
        System.out.println(predictions);
    }
}