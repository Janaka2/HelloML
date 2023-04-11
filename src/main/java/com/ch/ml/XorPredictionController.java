package com.ch.ml;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/predict/xor")
public class XorPredictionController {

    private static final Logger log = LoggerFactory.getLogger(XorPredictionController.class);
    private final MultiLayerNetwork model;

    public XorPredictionController() {
        log.info("Training a simple XOR model...");

        // Create and train the model
        DataSetIterator iterator = SimpleMLModel.createXorDataSetIterator();
        model = SimpleMLModel.createXorModel();
        SimpleMLModel.trainModel(model, iterator, 5000);
    }

    @PostMapping
    public double predict(@RequestBody double[] input) {
        INDArray inputData = Nd4j.create(input);
        INDArray output = model.output(inputData);
        double prediction = output.getDouble(0);

        log.info("Input: [{}, {}] -> Output: {}", input[0], input[1], prediction);
        return prediction;
    }
}
