import os
from pyexpat import model
from random import sample
import numpy as np
from numpy import genfromtxt
from os import path
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.mixture import GaussianMixture
import pandas as pd
from Models import ModelKNN, ModelLR, ModelMoG
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut


# File locations
dataFileName = '../Data/Genes/data.csv'
labelsFileName = '../Data/Genes/labels.csv'

# Storage of data to speed up debugging
rawDataFile = './PreProcessedData/rawData.npy'
labelsFile = './PreProcessedData/labels.npy'
labelsNameFile = './PreProcessedData/labelNames.npy'

LoadDataset = False

class Pipeline:
    def __init__(self):
        # The purpose of this function is to speed up debugging by not having to read the data set each run
        if (path.exists(rawDataFile)==False or path.exists(labelsFile)==False or path.exists(labelsNameFile)==False or LoadDataset): 
            # Read data
            self.rawData = genfromtxt(dataFileName, skip_header=True, delimiter=',')[:,1:] # shape is 802x20532 
            
            # Read and encode labels
            rawLabels = np.genfromtxt(labelsFileName, skip_header=True, delimiter=',',dtype=str)[:,1:] # shape is 802x2          
            encodedNames =[]
            encodedLabels = []
            for label in rawLabels:
                labelValue = label[0]
                if labelValue in encodedNames:
                    encodedLabel = encodedNames.index(labelValue)
                    encodedLabels.append(encodedLabel)
                else:
                    encodedLabels.append(len(encodedNames))
                    encodedNames.append(labelValue)      
            self.labelNames = np.array(encodedNames, dtype=str)
            self.labels  = np.array(encodedLabels)
            
            np.save(rawDataFile, self.rawData)
            np.save(labelsFile, self.labels)
            np.save(labelsNameFile, self.labelNames)
        else:
            self.rawData = np.load(rawDataFile)
            self.labels = np.load(labelsFile)
            self.labelNames = np.load(labelsNameFile, allow_pickle=True)

    def exploreData(self):
        # Exploration of the data

        # count number of samples per class
        samplesPerClass = [0]*len(self.labelNames)
        for label in self.labels:
            samplesPerClass[label] +=1
        leastItems = np.min(samplesPerClass)

        self.classWeights={}
        for index, i in enumerate(samplesPerClass):
            weight = leastItems/i
            self.classWeights[index]=weight
        
        print("The data set has "+ str(len(self.labels))+ " samples divided over " + str(len(self.labelNames)) + " classes.")
        print("For labels " + str(self.labelNames) + ",the number of samples per class are: " + str(samplesPerClass))
        print("Number of features per sample: "+ str(self.rawData.shape[1]))
        
        zeroColumns = 0
        for column in self.rawData.T:
            if len(column) == np.count_nonzero(column==0.0):
                zeroColumns +=1
        print("However, of those features, "+ str(zeroColumns)+ " features are zero for each sample")
        print()

    def splitData(self):
        testSetFactor = 0.1
        # Shuffle the data set using a seed
        p = np.random.RandomState(0).permutation(len(self.labels))
        rawDataX = self.rawData[p]
        self.dataY = self.labels[p]
        testSetCount = round(len(rawDataX)*testSetFactor)
        self.rawTrainX = rawDataX[0:len(rawDataX)-testSetCount]
        self.trainY = self.dataY[0:len(self.dataY)-testSetCount]
        self.rawTestX = rawDataX[len(rawDataX)-testSetCount:len(rawDataX)]
        self.testY = self.dataY[len(self.dataY)-testSetCount:len(self.dataY)]

    def preProcess(self):
        trainingMean = np.mean(self.rawTrainX, axis=0) 
        centeredTrainingData = self.rawTrainX - trainingMean
        trainingSD = np.std(centeredTrainingData, axis=0)
        zeroColumns = trainingSD == 0
        nonZeroSD = np.delete(trainingSD,zeroColumns,axis=0)
        self.trainX = np.delete(centeredTrainingData,zeroColumns,axis=1)/nonZeroSD

        # TODO: check if the way the mean is subtracted, and the standard deviation is correct
        testMean = trainingMean
        centeredTestingData = self.rawTestX - testMean
        self.testX = np.delete(centeredTestingData,zeroColumns,axis=1)/nonZeroSD

    def pcaSearch(self):
        n_comps = len(self.trainX)
        pca = PCA(n_components=n_comps)
        pca.fit(self.trainX)

        exp_var = pca.explained_variance_ratio_ * 100

        cum_exp_var = np.cumsum(exp_var)
        for i in range(len(cum_exp_var)-1):
                if (cum_exp_var[i] > 80):
                    optimalComponent = i+1
                    print(f"{optimalComponent} components leads to {cum_exp_var[i]} explained variance.")
                    break

        plt.bar(range(1, n_comps+1), exp_var, align='center',
                label='Individual explained variance')

        plt.step(range(1, n_comps+1), cum_exp_var, where='mid',
                label='Cumulative explained variance', color='red')

        plt.ylabel('Explained variance percentage')
        plt.xlabel('Principal component index')
        plt.xticks(ticks=np.arange(1,n_comps+1, 2))
        plt.legend(loc='best')
        plt.tight_layout()
        # plt.savefig("./Figures/Barplot.png")
        return optimalComponent
      
    def featureExtraction(self, numberOfComponents):
        self.pca = PCA(numberOfComponents)
        reducedDimensionsData = self.pca.fit_transform(self.trainX)
  
        if (numberOfComponents <= 3):
            fig = plt.figure(figsize = (8,8))
            if numberOfComponents == 3:
                ax = fig.add_subplot(projection='3d')
            else:
                ax = fig.add_subplot()
                
            for index, label in enumerate(self.labelNames):
                indicesOfClass = self.labels == index
                points = reducedDimensionsData[indicesOfClass] 
                
                if numberOfComponents == 1:
                    ax.hist(points, alpha=0.5)
                elif numberOfComponents == 2:
                    ax.scatter(points[0],points[1])
                else:
                    ax.scatter(points[0],points[1],points[2])
            plt.xlabel("Principal component 1")
            plt.ylabel("Principal component 2")
            plt.title("PCA")
            # plt.savefig(f"Figures{os.sep}PCA")
        return reducedDimensionsData
    
    def validation(self, model, data, labels):
        # TODO: for training cost sensitive error function becuase number of samples per class
        model.newModel()
        
        accuracies = []
        for fold in range(len(labels)):
            if fold % 100 == 0:
                print("Fold " + str(fold+1))
            trainX = np.delete(data,fold,axis=0)
            trainY = np.delete(labels,fold,axis=0)
            valX = data[[fold]]
            valY = labels[[fold]]
            
            # Train and validate the model
            model.train(trainX, trainY, self.classWeights)
            accuracies.append(model.test(valX, valY))

        accuracy = np.mean(accuracies)
        print("Done, average accuracy is: " + str(round(accuracy,3))+"%")
        return accuracy

if __name__=="__main__":
    pipeline = Pipeline()
    pipeline.exploreData()
    pipeline.splitData()
    pipeline.preProcess()
    estimatedComponents = pipeline.pcaSearch()

    trainingFeatures = pipeline.featureExtraction(estimatedComponents)
    print(len(trainingFeatures))
    input_data = trainingFeatures
    targets = pipeline.trainY

    cv_obj = LeaveOneOut()

    knn_model = KNeighborsClassifier()
    knn_parameters = {'n_neighbors':[1,3],'p':[1,2],'weights':('uniform', 'distance')} 
    knn_search = GridSearchCV(knn_model, knn_parameters, cv = cv_obj, verbose=1)
    knn_search.fit(input_data, targets)
    

    lr_model = LogisticRegression()
    lr_parameters = {'penalty':('l2', 'l1', 'elasticnet', 'none')}
    lr_search = GridSearchCV(lr_model, lr_parameters, cv = cv_obj, verbose=1)
    lr_search.fit(input_data, targets)

    print(f"With a validation accuracy of {knn_search.best_score_}%, the best combination of hyperparameter settings for KNN is:")
    print(knn_search.best_params_)
    print(f"With a validation accuracy of {lr_search.best_score_}%, the best combination of hyperparameter settings for Logistic Regression is:")
    print(lr_search.best_params_)

    # TODO: finally, evaluate all models on the test data
    reducedTestData = pipeline.pca.transform(pipeline.testX)