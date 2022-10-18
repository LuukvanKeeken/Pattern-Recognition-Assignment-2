import os
from random import sample
import numpy as np
from numpy import genfromtxt
from os import path
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

# File locations
dataFileName = '../Data/Genes/data.csv'
labelsFileName = '../Data/Genes/labels.csv'

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
        leastItems = np.min(samplesPerClass)

        self.classWeights={}
        for index, i in enumerate(samplesPerClass):
            weight = leastItems/i
            self.classWeights[index]=weight
        
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
            # plt.savefig(f"Figures{os.sep}PCA")
        return reducedDimensionsData
    
    def validation(self, data, labels, k):
        model = Model(k, 16)
        # TODO: for training cost sensitive error function becuase number of samples per class
        
        accuracies = []
        for fold in range(len(labels)):
            if fold % 100 == 0:
                print("Fold " + str(fold+1))
            trainX = np.delete(data,fold,axis=0)
            trainY = np.delete(labels,fold,axis=0)
            valX = data[[fold]]
            valY = labels[[fold]]
            
            # Train and validate the model
            model.train(trainX, trainY)
            accuracies.append(model.test(valX, valY))

        accuracy = np.mean(accuracies)
        print("Done, average validation accuracy is: " + str(round(accuracy,3))+"%")
        return accuracy

   
        

class Model:
    def __init__(self, k, seed):
        self.knn_model = KNeighborsClassifier(n_neighbors=k)

    def train(self, trainX, trainY):
        self.knn_model.fit(trainX, trainY)

    def test(self, testX, testY):
        predictY = self.knn_model.predict(testX)
        #cnf_matrix = metrics.confusion_matrix(testY, predictY)   
        #print(cnf_matrix)
        accuracy = testY==predictY
        # correct = 0
        # for test, predict in zip(testY, predictY):
        #     if (test==predict):
        #         correct +=1
        # accuracy = correct/len(testY)
        return accuracy


     
if __name__=="__main__":
    pipeline = Pipeline()
    pipeline.preProcess()
    pipeline.exploreData()


    pipeline.splitData()
    
    min_num_of_components = 600
    max_num_of_components = 602
    min_k = 1
    max_k = 15
    accuracies = []
    highest_acc = -1
    best_hyperparams = {}
    for n_components in range(min_num_of_components, max_num_of_components+1):
        dataReduced = pipeline.dimensionReduction(pipeline.trainX, pipeline.trainY, n_components)
        for k in range(min_k, max_k+1, 2):
            print("n_components = " + str(n_components) + ", k = " + str(k))
            accuracy = pipeline.validation(dataReduced, pipeline.trainY, k)
            accuracies.append(accuracy)#, kFolds=10))
            if accuracy > highest_acc:
                highest_acc = accuracy
                best_hyperparams = {'n_components' : n_components, 'k' : k}

    print(best_hyperparams)
    print(highest_acc)
    fig = plt.figure(figsize = (8,8))
    plt.plot(range(1,len(accuracies)+1), accuracies)
    
    plt.xlabel("Dimensions")
    plt.ylabel("Accuracy")
    plt.title("Accuracies with different dimensions")
    # plt.savefig(f"..{os.sep}Figures{os.sep}GridSearch")
