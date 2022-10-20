from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
import numpy as np
from Validator import Validator
import matplotlib.pyplot as plt

class GridSearch:

    def __init__(self,  validator):

        self.validator = validator
        self.results = {}
        self.CVresults = {}
        self.knn_results = {}
        self.svm_results = {}

    def gridSearch(self, classifier):

        if classifier == "knn":
        
            n_neighbors = np.arange(1 , 11, 1)
            distance_metrics = ["euclidean", "cosine" , "manhattan", "minkowski"]

            for i in n_neighbors:
                
                knn = KNeighborsClassifier(n_neighbors=i, weights='uniform')   # uniform weights. All points in each neighborhood are weighted equally.
                self.validator.crossValidation(knn, 2)
                #print("Nearest Neighbours: " + str(i))
                #self.validator.printCVResults()
                self.knn_results[i] = {'uniform' : self.validator.CVresults}

                for m in distance_metrics:
                    # Weight data points by the inverse of their distance. 
                    # Closer neighbors of a query point will have a greater influence than neighbors which are further away.
                    knn = KNeighborsClassifier(n_neighbors=i, weights='distance', metric=m) 
                    self.validator.crossValidation(knn, 2)
                    self.knn_results[i][m] = self.validator.CVresults


        elif classifier == "svm":
            
            C = np.arange(0.5, 2.1, 0.1)
            Kernels = ["linear", "poly", "rbf", "sigmoid", "precomputed"]

            for c in C:
                self.svm_results[c] = {}
                for k in Kernels:

                    SVM = svm.SVC(C = c, kernel = k)
                    self.validator.crossValidation(SVM, 2)
                    self.svm_results[c][k] = self.validator.CVresults


    
    def plotKnnResults(self, eval_metric):
        
        X = list(self.knn_results.keys())
        results = {'uniform' : [], 'euclidean' : [], 'cosine' : [], 'manhattan' : [], 'minkowski' : []}
        for n in X:
            current_dict = self.knn_results[n]
            for metric in current_dict.keys():
                results[metric].append(current_dict[metric][eval_metric])

        fig = plt.figure(figsize=(10, 8), dpi = 80)
        for i in results.keys():
            plt.plot(X, results[i], label = i, marker='o' )
       
    
        plt.title("{} per k and distance metric".format(eval_metric), fontsize=18)
        plt.xlabel("Number of nearest neighbors", fontsize=15)
        plt.ylabel(eval_metric, fontsize=15)
        plt.legend(fontsize = 12)
        plt.xticks(X)
        plt.grid()

        plt.show()



    def plotSVMResults(self, eval_metric):
        
        X = list(self.svm_results.keys())
        results = {'linear' : [], 'poly' : [], 'rbf' : [], 'sigmoid' : [], 'precomputed' : []}
        for n in X:
            current_dict = self.knn_results[n]
            for kernel in current_dict.keys():
                results[kernel].append(current_dict[kernel][eval_metric])

        fig = plt.figure(figsize=(10, 8), dpi = 80)
        for i in results.keys():
            plt.plot(X, results[i], label = i, marker='o' )


        plt.title("{} per C and kernel type".format(eval_metric), fontsize=18)
        plt.xlabel("C", fontsize=15)
        plt.ylabel(eval_metric, fontsize=15)
        plt.legend(fontsize = 12)
        plt.xticks(X)
        plt.grid()

        plt.show()

    

