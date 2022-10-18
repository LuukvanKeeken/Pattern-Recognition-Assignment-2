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
reProcessPreprocessedData = True

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
            nanColumns = np.isnan(self.preProcessedData[0])
            self.preProcessedData = np.delete(self.preProcessedData,nanColumns,axis=1)
            #self.preProcessedData[np.isnan(self.preProcessedData)] = 0.0

            # Create correlation matrix
            # df = pd.DataFrame( self.preProcessedData)#, columns = ['Column_A','Column_B','Column_C'])
            # #corr_matrix = df.corr().abs()
            # corr_matrix = np.corrcoef(self.preProcessedData,)
            # #corr_matrix = self.preProcessedData.corrcoef()

            # # Select upper triangle of correlation matrix
            # upper = corr_matrix[(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))]

            # # Find features with correlation greater than 0.95
            # to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

            # # Drop features 
            # df.drop(to_drop, axis=1, inplace=True)
            # self.preProcessedData = df.to_numpy()

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
        testSetFactor = 0.2
        # Shuffle the data set using a seed
        p = np.random.RandomState(0).permutation(len(self.preProcessedData))
        self.dataX = self.preProcessedData[p]
        self.dataY = self.preProcessedLabels[p]
        testSetCount = round(len(self.dataX)*testSetFactor)
        self.trainX = self.dataX[0:len(self.dataX)-testSetCount]
        self.trainY = self.dataY[0:len(self.dataY)-testSetCount]
        self.testX = self.dataX[len(self.dataX)-testSetCount:len(self.dataX)]
        self.testY = self.dataY[len(self.dataY)-testSetCount:len(self.dataY)]


    def dimensionReductionTestData(self, data):
        reducedTestData = self.pca.transform(data)
        return reducedTestData

    def dimensionReduction(self, data, labels, numberOfComponents):
        self.pca = PCA(numberOfComponents)
        reducedDimensionsData = self.pca.fit_transform(data)
        
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
    
    def validation(self, data, labels):
        model = Model()
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
            model.train(trainX, trainY, self.classWeights)
            accuracies.append(model.test(valX, valY))

        accuracy = np.mean(accuracies)
        print("Done, average accuracy is: " + str(round(accuracy,3))+"%")
        return accuracy

    def clustering(self, datasetX, datasetY):
        #datasetX = self.trainX
        #datasetY = self.trainY
        
        AIC = []
        gmms = []
        componentsRange = range(5,15,1)
        for components in componentsRange:
            np.random.seed(42)
            newGmm = GaussianMixture(n_components=components, random_state=42)
                                    # covariance_type='full', 
                                    # tol=1e-10,
                                    # reg_covar=1e-10,
                                    # max_iter=100, 
                                    # n_init=20,
                                    # init_params='kmeans', 
                                    # weights_init=None, 
                                    # means_init=None, 
                                    # precisions_init=None, 
                                    # random_state=None, 
                                    # warm_start=False, 
                                    # verbose=0,
                                    # verbose_interval=10)
            newGmm.fit(datasetX) 
            AIC.append(newGmm.aic(datasetX))
            gmms.append(newGmm)
            #BIC = gmm.big(datasetX)

        plt.close()
        plt.cla()
        plt.clf()
        fig = plt.figure(figsize = (8,8))
        plt.plot(componentsRange,AIC)
        
        plt.xlabel("Components")
        plt.ylabel("AIC")
        plt.title("AIC of MoG")
        plt.savefig(f"Figures{os.sep}MoG")



        gmm = gmms[np.argmin(AIC)]
        gmm.n_components
        finalComponents = gmm.n_components #np.argmin(AIC)

        
        #gmm = GaussianMixture(n_components=finalComponents, random_state=123)
        #gmm.fit(datasetX) 

        
        # Match the labels of the MoG with the real labels. 
        labels = np.zeros((finalComponents, 5))
        for i in range(len(datasetY)):
            realLabel = datasetY[i]
            image = datasetX[i]
            image = image.reshape(1,-1)
            classPredictions = int(gmm.predict(image))
            labels[classPredictions][realLabel] += 1
        modelLabels = [0] * finalComponents
        good = 0
        for index in range(finalComponents):
            mostCount = np.argmax(labels[index])
            good += labels[index][mostCount]
            modelLabels[index] = mostCount
        
        # calculate the accuracy of the trained model on the training data
        accuracy = good/len(datasetY)

        return gmm, modelLabels

    def predictMoG(self, gmm, modelLabels, testData, testLabels):
        correct = 0
        for sample, label in zip(testData,testLabels):
            sample = sample.reshape(1,-1)
            prediction = modelLabels[int(gmm.predict(sample))]
            if prediction == label:
                correct+=1
        accuracy = correct/len(testLabels)
        return accuracy

    # def validation(self, data, labels, kFolds):
    #     model = Model()
    #     # TODO: for training cost sensitive error function becuase number of samples per class
    #     kFolds = 800
    #     foldSize = 1# round(len(data)/kFolds)
    #     accuracies = []
    #     for fold in range(kFolds):
    #         print("Fold " + str(fold+1))
    #         # Split the data in training and validation data
    #         valSetIdx = np.arange(fold*foldSize,(fold+1)*foldSize, 1, int)
    #         trainX = np.delete(data, valSetIdx,axis=0)
    #         trainY = np.delete(labels, valSetIdx,axis=0)
    #         valX = data[valSetIdx]
    #         valY = labels[valSetIdx]
            
    #         # Train and validate the model
    #         model.train(trainX, trainY)
    #         accuracies.append(model.test(valX, valY))

    #     accuracy = np.mean(accuracies)
    #     print("Done, average accuracy is: " + str(round(accuracy,3))+"%")
    #     return accuracy
        

class Model:
    def __init__(self):
        self.regressionModel = LogisticRegression(random_state=16)

    def train(self, trainX, trainY, classWeights):
        self.regressionModel = LogisticRegression(random_state=16, max_iter=10000, class_weight=classWeights)
        self.regressionModel.fit(trainX, trainY)

    def test(self, testX, testY):
        predictY = self.regressionModel.predict(testX)
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
    #mmg, modelLabels = pipeline.clustering(pipeline.testX, pipeline.testY)
    dataReducedTraining = pipeline.dimensionReduction(pipeline.trainX, pipeline.trainY, 20)


    mmg, modelLabels = pipeline.clustering(dataReducedTraining, pipeline.trainY)

    dataReducedTesting = pipeline.dimensionReductionTestData(pipeline.testX)
    pipeline.predictMoG(mmg, modelLabels, dataReducedTesting, pipeline.testY)

    maxNumberOfDimensions = 10
    accuracies = []
    for dimension in range(1,maxNumberOfDimensions+1):
        print("Grid search "+ str(dimension))
        dataReduced = pipeline.dimensionReduction(pipeline.dataX, pipeline.dataY, dimension)
        accuracies.append(pipeline.validation(dataReduced, pipeline.dataY))#, kFolds=10))

    fig = plt.figure(figsize = (8,8))
    plt.plot(range(1,len(accuracies)+1), accuracies)
    
    plt.xlabel("Dimensions")
    plt.ylabel("Accuracy")
    plt.title("Accuracies with different dimensions")
    plt.savefig(f"Figures{os.sep}GridSearch")
