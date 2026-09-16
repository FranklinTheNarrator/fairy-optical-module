package org.phaethon;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

import java.io.File;
import java.io.IOException;
import java.util.Random;

public class DatasetLoader {
    private static final int HEIGHT = 128;
    private static final int WIDTH = 128;
    private static final int CHANNELS = 3;

    private static final int NUM_CLASSES = 2;
    private static final int BATCH_SIZE = 64;
    private static final long SEED = 123;

    public static DataSetIterator loadTrainData() throws IOException {
        File trainDir = new File("data/train");

        if (!trainDir.exists()) {
            throw new IOException("Path data/train was not found!");
        }

        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();

        FileSplit fileSplit = new FileSplit(trainDir, NativeImageLoader.ALLOWED_FORMATS, new Random(SEED));

        ImageRecordReader recordReader = new ImageRecordReader(HEIGHT, WIDTH, CHANNELS, labelMaker);
        recordReader.initialize(fileSplit);

        DataSetIterator iterator = new RecordReaderDataSetIterator(
                recordReader, BATCH_SIZE, 1, NUM_CLASSES
        );

        ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);
        scaler.fit(iterator);
        iterator.setPreProcessor(scaler);

        return iterator;
    }

    static void main(String[] args) throws IOException {
        System.out.println("Uploading data...");
        DataSetIterator trainIter = loadTrainData();

        if (trainIter.hasNext()) {
            var batch = trainIter.next();
            System.out.println("Features: " + batch.getFeatures().shapeInfoToString());
            System.out.println("Labels: " + batch.getLabels().shapeInfoToString());
        }

        System.out.println("All good!");
    }
}