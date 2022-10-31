
from distutils.command.build import build
from turtle import distance
from sklearn.model_selection import train_test_split
from DataHandler import DataHandler
from SIFTFeatureExtractor import SIFTFeatureExtractor
import cv2
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from GridSearch import GridSearch
from SIFT import SIFT
import sys
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm, tree, ensemble
from sklearn.metrics import f1_score, silhouette_score
from sklearn.cluster import KMeans

def main():
    hyperparameters = {
        "unreduced": {
            "clusters_2": 5,
            "rf_metric": "gini",
            "estimators": 170,
            "distance": "cosine",
            "kernel": "rbf",
            "C": 1.8,
            "K": 10,
            "weights": "uniform"
        },
        "reduced": {
            "keypoints": {
                "svm": 35,
                "rf": 35,
                "knn": 40,
                "km": 35
            },
            "clusters_1": {
                "svm": 25,
                "rf": 35,
                "knn": 45,
                "km": 25,
            },
            "clusters_2": 5,
            "rf_metric": "gini",
            "estimators": 140,
            "distance": "cosine",
            "kernel": "poly",
            "C": 0.7,
            "K": 16,
            "weights": "uniform"
        },
        "augmented": {
            "keypoints": {
                "svm": 25,
                "rf": 35,
                "knn": 25,
                "km": 25,
            },
            "clusters_1": {
                "svm": 40,
                "rf": 30,
                "knn": 25,
                "km": 40,
            },
            "clusters_2": 5,
            "rf_metric": "entropy",
            "estimators": 200,
            "distance": "manhattan",
            "kernel": "rbf",
            "C": 2,
            "K": 10,
            "weights": "uniform"
        },
    }

    data_type = sys.argv[1]
    model_type = sys.argv[2]

    # Load and process Data
    data_dir = f"BigCats{os.sep}"
    dh = DataHandler()
    dh.loadData(data_dir)
    dh.plotClassDistribution()
    dh.showExampleImages()
    target_dict = dh.convertLabelsToNumeric()
    images = dh.preprocessData(data_type)

    # Make Train - Test split
    X_train, X_test, y_train, y_test = train_test_split(
        images, dh.class_labels, test_size=0.2, shuffle=True, random_state=0
    )

    if data_type == "augmented":
        X_train = np.append(X_train, dh.augmentData(X_train), axis=0)
        y_train += y_train

    if data_type != "unreduced":
        # Feature Extraction with optimal params
        feature_extractor = SIFTFeatureExtractor()
        descriptors_train = feature_extractor.build_descriptors(
            X_train, hyperparameters[data_type]['keypoints'][model_type], "descriptors_train.npy")
        descriptors_test = feature_extractor.build_descriptors(
            X_test, hyperparameters[data_type]['keypoints'][model_type], "descriptors_test.npy")

        vocab = feature_extractor.build_vocab(
            descriptors_train, hyperparameters[data_type]['clusters_1'][model_type])
        train_histograms = feature_extractor.build_histograms(
            descriptors_train, y_train, hyperparameters[data_type]['keypoints'][model_type], vocab)
        test_histograms = feature_extractor.build_histograms(
            descriptors_test, y_test, hyperparameters[data_type]['keypoints'][model_type], vocab, filename="histograms_test")

        X_train, y_train = ([hist[0] for hist in train_histograms], [hist[1]
                                                                     for hist in train_histograms])                                       
        X_test, y_test = ([hist[0] for hist in test_histograms], [hist[1]
                                                                  for hist in test_histograms])

    # Training and Testing the Models with optimal params
    if model_type == "svm":
        svm_model = svm.SVC(C=hyperparameters[data_type]['C'],
                            kernel=hyperparameters[data_type]['kernel'])
        svm_model.fit(X_train, y_train)
        print(f"Accuracy is: {svm_model.score(X_test, y_test)}")
        print(
            f"F1 is: {f1_score(y_test, svm_model.predict(X_test), average='macro')}")
        
        dh.plotConfusionMatrix(model_type, data_type, svm_model.predict(X_test), y_test)
    elif model_type == "rf":
        rf = ensemble.RandomForestClassifier(
            n_estimators=hyperparameters[data_type]["estimators"], criterion=hyperparameters[data_type]['rf_metric'], random_state=0)
        rf.fit(X_train, y_train)
        print(f"Accuracy is: {rf.score(X_test, y_test)}")
        print(
            f"F1 is: {f1_score(rf.predict(X_test), y_test, average='macro')}")
        dh.plotConfusionMatrix(model_type, data_type, rf.predict(X_test), y_test)
    elif model_type == "knn":
        knn = KNeighborsClassifier(
            n_neighbors=hyperparameters[data_type]['K'], weights=hyperparameters[data_type]["weights"])
        knn.fit(X_train, y_train)
        print(f"Accuracy is: {knn.score(X_test, y_test)}")
        print(
            f"F1 is: {f1_score(knn.predict(X_test), y_test, average='macro')}")
        
        dh.plotConfusionMatrix(model_type, data_type, knn.predict(X_test), y_test)
    elif model_type == "km":
        km = KMeans(n_clusters=hyperparameters[data_type]['clusters_2'],
                    random_state=0)
        X = np.append(X_train, X_test, axis=0)
        km.fit_predict(X)
        print(f"Silhouette score is: {silhouette_score(X, km.labels_, metric='euclidean')}")


if __name__ == "__main__":
    main()
