
from distutils.command.build import build
from turtle import distance
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist
from DataHandler import DataHandler
from SIFTFeatureExtractor import SIFTFeatureExtractor
from Models import SVM
import cv2
import os
import numpy as np
import pickle
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

def main():
    N_KEYPOINTS = 10
    data_dir = f"BigCats{os.sep}"
    dh = DataHandler()
    dh.loadData(data_dir)

    dh.showClassDistribution()
    dh.showExampleImages()

    target_dict = dh.convertLabelsToNumeric()
    images = dh.preprocessData()

    # Train - Test split
    X_train, X_test, y_train, y_test = train_test_split(
        images, dh.class_labels, test_size=0.2, shuffle=True, random_state=0
    )

    feature_extractor = SIFTFeatureExtractor()
    descriptors_train = feature_extractor.build_descriptors(X_train, N_KEYPOINTS, "descriptors_train.npy")
    descriptors_test = feature_extractor.build_descriptors(X_test, N_KEYPOINTS, "descriptors_test.npy")

    vocab = feature_extractor.build_vocab(descriptors_train)

    train_histograms = feature_extractor.build_histograms(
        descriptors_train, y_train, N_KEYPOINTS, vocab)
    test_histograms = feature_extractor.build_histograms(
        descriptors_test, y_test, N_KEYPOINTS, vocab, filename="histograms_test")

    # Training and Testing the Model
    model = SVM()
    model.train([hist[0] for hist in train_histograms], [hist[1]
                for hist in train_histograms])
    model.test([hist[0] for hist in test_histograms], [hist[1]
               for hist in test_histograms])


if __name__ == "__main__":
    main()
