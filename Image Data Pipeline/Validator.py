from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
import numpy as np


class Validator:
    def __init__(self,  train_X, train_y):
        self.train_X = train_X
        self.train_y = train_y
        self.results = {}
        self.CVresults = {}

    def cross_validation(self, classifier, k_folds=5):
        """Perform K-fold cross validation"""
        self.results = cross_val_score(
            classifier, self.train_X, self.train_y, cv=k_folds, scoring='f1_micro')
        self.CVresults = np.mean(self.results)

    def print_cv_results(self):
        """Print the final results for cross validation"""
        print("Fit time: " + str(np.mean(self.results["fit_time"])))
        print("Accuracy: " + str(np.mean(self.results["test_accuracy"])))
        print("Precision: " +
              str(np.mean(self.results["test_precision_micro"])))
        print("Recall: " + str(np.mean(self.results["test_recall_micro"])))
        print("F1: " + str(np.mean(self.results["test_f1_micro"])))
