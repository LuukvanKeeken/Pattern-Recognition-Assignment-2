from sklearn.model_selection import cross_validate
import numpy as np

class Validator:

    def __init__(self,  train_X, train_y):
      
        self.train_X = train_X
        self.train_y = train_y
        self.results = {}
        self.CVresults = {}

    
    def crossValidation(self, classifier, k_folds=5):


        scoring = ["accuracy", "precision_micro", "recall_micro", "f1_micro"]
        results = cross_validate(classifier, self.train_X, self.train_y,
                                    cv = k_folds,
                                    scoring= scoring)

        self.results = results

        self.CVresults = {
                            "Accuracy" : np.mean(self.results["test_accuracy"]),
                            "Precision" : np.mean(self.results["test_precision_micro"]),
                            "Recall" : np.mean(self.results["test_recall_micro"]),
                            "F1" : np.mean(self.results["test_f1_micro"])
                         }

    def printCVResults(self):
       
        print("Fit time: " + str(np.mean(self.results["fit_time"])))
        print("Accuracy: " + str(np.mean(self.results["test_accuracy"])))
        print("Precision: " + str(np.mean(self.results["test_precision_micro"])))
        print("Recall: " + str(np.mean(self.results["test_recall_micro"])))
        print("F1: " + str(np.mean(self.results["test_f1_micro"])))


