import os
from select import select
import numpy as np
from numpy import genfromtxt
from os import path
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

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
        if (path.exists(labelsFile)==False or path.exists(rawDataFile)==False or path.exists(rawLabelsFile)==False or reProcessRawData):
            self.labelsDict =[]
            self.rawData = genfromtxt(dataFileName, skip_header=True, delimiter=',')[:,1:] # shape is 802x20532 

            # Read the labels files, and assign a class number to each label text
            readLabels = np.genfromtxt(labelsFileName, skip_header=True, delimiter=',',dtype=str)[:,1:] # shape is 802x2
            self.rawLabels = []
            dictIndex = 0
            for label in readLabels:
                labelValue = label[0]
                if labelValue in self.labelsDict:
                    encodedLabel = self.labelsDict.index(labelValue)
                    self.rawLabels.append(encodedLabel)
                else:
                    self.rawLabels.append(len(self.labelsDict))
                    self.labelsDict.append(labelValue)  
            self.labelsDict = np.array(self.labelsDict, dtype=str)
            np.save(labelsFile, self.labelsDict)
            np.save(rawDataFile, self.rawData)
            np.save(rawLabelsFile, self.rawLabels)
        else:
            self.rawData = np.load(rawDataFile)
            self.labelsDict= np.load(labelsFile, allow_pickle=True)
            self.rawLabels = np.load(rawLabelsFile)
    
    def preProcess(self):
        if (path.exists(preProcessedDataFile)==False or path.exists(preProcessedLabelsFile)==False or reProcessPreprocessedData):
            self.preProcessedData = self.rawData - np.mean(self.rawData, axis=0) 
            self.preProcessedData /= np.std(self.preProcessedData, axis=0)
            # when a column is zero, dividing by std is /0, which is not a number (nan). Replace them by 0.0
            self.preProcessedData[np.isnan(self.preProcessedData)] = 0.0
            self.preProcessedLabels = self.rawLabels
            np.save(preProcessedDataFile, self.preProcessedData)
            np.save(preProcessedLabelsFile, self.preProcessedLabels)
        else:
            self.preProcessedData = np.load(preProcessedDataFile)
            self.preProcessedLabels = np.load(preProcessedLabelsFile)

    def reduceDimension(self):
        numberOfComponents=1
        pca = PCA(numberOfComponents)
        self.reducedDimensionsData = pca.fit_transform(self.preProcessedData)
        
        if (numberOfComponents <= 3):
            fig = plt.figure(figsize = (8,8))
            if numberOfComponents == 3:
                ax = fig.add_subplot(projection='3d')
            else:
                ax = fig.add_subplot()
                
            for index, label in enumerate(self.labelsDict):
                indexesOfClass = self.preProcessedLabels == index
                points = self.reducedDimensionsData[indexesOfClass] 
                
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
    
    def splitData(self):
        testSetFactor = 0.1

        # Shuffle the data set using a seed
        p = np.random.RandomState(0).permutation(len(self.reducedDimensionsData))
        dataX = self.reducedDimensionsData[p]
        dataY = self.preProcessedLabels[p]
        testSetCount = len(dataX)*testSetFactor
        self.trainX = dataX[0:len(dataX)-testSetCount]
        self.trainY = dataY[0:len(dataY)-testSetCount]
        self.testX = dataX[len(dataX)-testSetCount,len(dataX)]
        self.testY = dataY[len(dataY)-testSetCount,len(dataY)]

    def crossValidate(self):
        model = Model()
        kFolds = 10
        # TODO: for training cost sensitive error function becuase number of samples per class
        foldSize = len(self.trainX)/kFolds
        errors = []
        for fold in range(kFolds):
            print("Fold " + str(fold+1))
            # Split the data in training and validation data
            valSetIdx = np.arange(fold*foldSize,(fold+1)*foldSize, 1, int)
            trainX = np.delete(self.trainX, valSetIdx,axis=0)
            trainY = np.delete(self.trainY, valSetIdx,axis=0)
            valX = self.trainX[valSetIdx]
            valY = self.trainY[valSetIdx]
            
            # Train and validate the model
            model.train(trainX, trainY, 100)
            errors.append(model.test(valX, valY))
        

class Model:
    def __init__():
        pass
    def train(self):
        pass
    def test(self):
        pass

     
if __name__=="__main__":
    pipeline = Pipeline()
    pipeline.preProcess()
    pipeline.reduceDimension()
    pipeline.splitData()