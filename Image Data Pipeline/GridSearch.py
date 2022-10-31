from SIFTFeatureExtractor import SIFTFeatureExtractor
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm, tree, ensemble
import numpy as np
from Validator import Validator
import matplotlib.pyplot as plt
import json


class GridSearch:
    def __init__(self,  X_train, y_train, data_type):
        self.X_train = X_train
        self.y_train = y_train
        self.data_type = data_type
        self.results = {}
        self.CVresults = {}
        self.knn_results = {}
        self.svm_results = {}
        self.rf_results = {}

    def gridsearch(self):
        """Run a full gridsearch for all models and save the results to file"""
        sift = SIFTFeatureExtractor()
        n_keypoints = range(5, 50, 5)
        n_clusters = range(5, 50, 5)

        if self.data_type != "unreduced":
            for i in n_keypoints:
                for j in n_clusters:
                    train_hists, test_hists = sift.extract_features(
                        self.X_train, self.y_train, n_keypoints=i, n_clusters=j)
                    validator = Validator([hist[0] for hist in train_hists], [
                        hist[1] for hist in train_hists])
                    self.run_knn(validator, i, j)
                    self.run_svm(validator, i, j)
                    self.run_rf(validator, i, j)
        else:
            validator = Validator(self.X_train, self.y_train)
            self.run_knn(validator)
            self.run_svm(validator)
            self.run_rf(validator)

    def run_knn(self, validator, keypoints="None", clusters="None"):
        """Run gridsearch for the K-nearest neighbour classifier"""
        n_neighbors = range(1, 31, 1)
        distance_metrics = ["euclidean",
                            "cosine", "manhattan", "minkowski"]
        for k in n_neighbors:
            # uniform weights. All points in each neighborhood are weighted equally.
            knn = KNeighborsClassifier(
                n_neighbors=k, weights='uniform')
            validator.cross_validation(knn, 2)
            update_dict = {
                keypoints: {
                    clusters: {
                        k: {
                            'uniform': validator.CVresults}}}}
            self.knn_results.update(update_dict)
            for m in distance_metrics:
                # Weight data points by the inverse of their distance.
                # Closer neighbors of a query point will have a greater influence than neighbors which are further away.
                knn = KNeighborsClassifier(
                    n_neighbors=k, weights='distance', metric=m)
                validator.cross_validation(knn, 5)
                self.knn_results[keypoints][clusters][k][m] = validator.CVresults
            f = open(f"knn_gs_{self.data_type}.json", "a")
            f.write(json.dumps(self.knn_results))
            f.close()

    def run_svm(self, validator, keypoints="None", clusters="None"):
        """Run gridsearch for the svm classifier"""
        c_range = np.arange(0.5, 2.1, 0.1)
        Kernels = ["linear", "poly", "rbf", "sigmoid"]

        for c in c_range:
            update_dict = {
                keypoints: {
                    clusters: {
                        str(c):
                            {}}}}
            self.svm_results.update(update_dict)
            for k in Kernels:
                SVM = svm.SVC(C=c, kernel=k)
                validator.cross_validation(SVM, 5)
                self.svm_results[keypoints][clusters][str(
                    c)][k] = validator.CVresults
            f = open(f"svm_gs_{self.data_type}.json", "a")
            f.write(json.dumps(self.svm_results))
            f.close()

    def run_rf(self, validator, keypoints="None", clusters="None"):
        """Run gridsearch for the random forest classifier"""
        criterions = ["gini", "entropy"]
        n_estimators = range(10, 210, 10)

        for n in n_estimators:
            update_dict = {
                keypoints: {
                    clusters: {
                        n: {
                        }}}}

            self.rf_results.update(update_dict)
            for c in criterions:
                RF = ensemble.RandomForestClassifier(
                    n_estimators=n, criterion=c)
                validator.cross_validation(RF, 5)
                self.rf_results[keypoints][clusters][n][c] = validator.CVresults
            f = open(f"rf_gs_{self.data_type}.json", "a")
            f.write(json.dumps(self.rf_results))
            f.close()
