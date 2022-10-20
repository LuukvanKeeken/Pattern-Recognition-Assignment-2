from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from Validator import Validator
import matplotlib.pyplot as plt

class GridSearch:

    def __init__(self, classifier, validator):
        if classifier:
            self.classifier = classifier
        else: 
            self.classifier = 'knn'

        self.validator = validator
        self.results = {}
        self.CVresults = {}

    def gridSearch(self, parameters = {}):

        if self.classifier == "knn":

            if parameters == {}:
                # If parameters were not supplied, proceed with predefined parameter setup
                n_neighbors = np.arange(1 , 11, 1)
                
                for i in n_neighbors:
                    
                    knn = KNeighborsClassifier(n_neighbors=i)
                    self.validator.crossValidation(knn, 5)
                    print("Nearest Neighbours: " + str(i))
                    self.validator.printCVResults()
                    
        
        elif self.classifier == "svn":
            pass

        elif self.classifier == "naive_bayes":
            pass