import os
from random import sample
import numpy as np
from numpy import genfromtxt
from os import path
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# File locations
dataFileName = './Data/Genes/data.csv'
labelsFileName = './Data/Genes/labels.csv'

# Storage of data to speed up debugging
labelsFile = './PreProcessedData/labels.npy'
rawDataFile = './PreProcessedData/rawData.npy'
rawLabelsFile = './PreProcessedData/rawLabels.npy'
preProcessedDataFile = './PreProcessedData/preProcessedData.npy'
preProcessedLabelsFile = './PreProcessedData/preProcessedLabels.npy'

reProcessRawData = False
reProcessPreprocessedData = False

class Pipeline:
    def __init__(self):
        # The purpose of this function is to speed up debugging by not having to read the data set each run
        if (path.exists(rawDataFile)==False or path.exists(rawLabelsFile)==False or reProcessRawData): 
            self.rawData = genfromtxt(dataFileName, skip_header=True, delimiter=',')[:,1:] # shape is 802x20532 
            self.rawLabels = np.genfromtxt(labelsFileName, skip_header=True, delimiter=',',dtype=str)[:,1:] # shape is 802x2
            np.save(rawDataFile, self.rawData)
            np.save(rawLabelsFile, self.rawLabels)
        else:
            self.rawData = np.load(rawDataFile)
            self.rawLabels = np.load(rawLabelsFile)
            
    def preProcess(self):
        if (path.exists(labelsFile)==False or path.exists(preProcessedDataFile)==False or path.exists(preProcessedLabelsFile)==False or reProcessPreprocessedData):
            # Note that preprocessing the data in this stage affects the test data later on as well. 
            self.preProcessedData = self.rawData - np.mean(self.rawData, axis=0) 
            self.preProcessedData /= np.std(self.preProcessedData, axis=0)
            # when a column is zero, dividing by std is /0, which is not a number (nan). Replace them by 0.0
            self.preProcessedData[np.isnan(self.preProcessedData)] = 0.0

            # Process the labels and encode them
            self.labelsDict =[]
            self.preProcessedLabels = []
            for label in self.rawLabels:
                labelValue = label[0]
                if labelValue in self.labelsDict:
                    encodedLabel = self.labelsDict.index(labelValue)
                    self.preProcessedLabels.append(encodedLabel)
                else:
                    self.preProcessedLabels.append(len(self.labelsDict))
                    self.labelsDict.append(labelValue)  
            self.labelsDict = np.array(self.labelsDict, dtype=str)
            self.preProcessedLabels = np.array(self.preProcessedLabels)
            
            # Save pre processed data, labels and the label encodings
            np.save(preProcessedDataFile, self.preProcessedData)
            np.save(preProcessedLabelsFile, self.preProcessedLabels)
            np.save(labelsFile, self.labelsDict)
        else:
            self.preProcessedData = np.load(preProcessedDataFile)
            self.preProcessedLabels = np.load(preProcessedLabelsFile)
            self.labelsDict= np.load(labelsFile, allow_pickle=True)

    def exploreData(self):
        # Exploration of the data

        # count number of samples per class
        samplesPerClass = [0]*len(self.labelsDict)
        for label in self.preProcessedLabels:
            samplesPerClass[label] +=1
        
        print("The data set has "+ str(len(self.preProcessedData))+ " samples divided over " + str(len(self.labelsDict)))
        print("For labels " + str(self.labelsDict) + ",the number of classes are: " + str(samplesPerClass))
        print("Number of features per sample: "+ str(self.preProcessedData.shape[1]))
        
        zeroColumns = 0
        for column in self.rawData.T:
            if len(column) == np.count_nonzero(column==0.0):
                zeroColumns +=1
        print("However, of those features, "+ str(zeroColumns)+ " features are zero for each sample")


    def splitData(self):
        testSetFactor = 0.1
        # Shuffle the data set using a seed
        p = np.random.RandomState(0).permutation(len(self.preProcessedData))
        self.dataX = self.preProcessedData[p]
        self.dataY = self.preProcessedLabels[p]
        testSetCount = round(len(self.dataX)*testSetFactor)
        self.trainX = self.dataX[0:len(self.dataX)-testSetCount]
        self.trainY = self.dataY[0:len(self.dataY)-testSetCount]
        self.testX = self.dataX[len(self.dataX)-testSetCount:len(self.dataX)]
        self.testY = self.dataY[len(self.dataY)-testSetCount:len(self.dataY)]


    def dimensionReduction(self, data, labels, numberOfComponents):
        pca = PCA(numberOfComponents)
        reducedDimensionsData = pca.fit_transform(data)
        
        if (numberOfComponents <= 3):
            fig = plt.figure(figsize = (8,8))
            if numberOfComponents == 3:
                ax = fig.add_subplot(projection='3d')
            else:
                ax = fig.add_subplot()
                
            for index, label in enumerate(self.labelsDict):
                indicesOfClass = labels == index
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
            plt.savefig(f"Figures{os.sep}PCA")
        return reducedDimensionsData
    


    def validation(self, data, labels, kFolds):
        model = Model()
        # TODO: for training cost sensitive error function becuase number of samples per class
        foldSize = round(len(data)/kFolds)
        accuracies = []
        for fold in range(kFolds):
            print("Fold " + str(fold+1))
            # Split the data in training and validation data
            valSetIdx = np.arange(fold*foldSize,(fold+1)*foldSize, 1, int)
            trainX = np.delete(data, valSetIdx,axis=0)
            trainY = np.delete(labels, valSetIdx,axis=0)
            valX = data[valSetIdx]
            valY = labels[valSetIdx]
            
            # Train and validate the model
            model.train(trainX, trainY)
            accuracies.append(model.test(valX, valY))

        accuracy = np.mean(accuracies)
        print("Done, average accuracy is: " + str(round(accuracy,3))+"%")
        return accuracy
        

class Model:
    def __init__(self):
        self.regressionModel = LogisticRegression(random_state=16)

    def train(self, trainX, trainY):
        self.regressionModel = LogisticRegression(random_state=16, max_iter=10000)
        self.regressionModel.fit(trainX, trainY)

    def test(self, testX, testY):
        predictY = self.regressionModel.predict(testX)
        cnf_matrix = metrics.confusion_matrix(testY, predictY)   
        print(cnf_matrix)

        correct = 0
        for test, predict in zip(testY, predictY):
            if (test==predict):
                correct +=1
        accuracy = correct/len(testY)
        return accuracy


     
if __name__=="__main__":
    pipeline = Pipeline()
    pipeline.preProcess()
    pipeline.exploreData()


    pipeline.splitData()
    
    maxNumberOfDimensions = 10
    accuracies = []
    for dimension in range(1,maxNumberOfDimensions+1):
        print("Grid search "+ str(dimension))
        dataReduced = pipeline.dimensionReduction(pipeline.dataX, pipeline.dataY, dimension)
        accuracies.append(pipeline.validation(dataReduced, pipeline.dataY, kFolds=10))

    fig = plt.figure(figsize = (8,8))
    plt.plot(range(1,len(accuracies)+1), accuracies)
    
    plt.xlabel("Dimensions")
    plt.ylabel("Accuracy")
    plt.title("Accuracies with different dimensions")
    plt.savefig(f"Figures{os.sep}GridSearch")
