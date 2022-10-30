import sys
from os import path
import pickle
import numpy as np
from numpy import genfromtxt
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE
from sklearn.metrics import silhouette_score
from sklearn.base import clone
from fcmeans import FCM


# Data set locations
dataFileName = './Data/Genes/data.csv'
labelsFileName = './Data/Genes/labels.csv'

# Storage of data to speed up debugging
rawDataFile = './Gene Data Pipeline/Data/rawData.npy'
labelsFile = './Gene Data Pipeline/Data/labels.npy'
labelsNameFile = './Gene Data Pipeline/Data/labelNames.npy'
GridSearchResultsFile = './Gene Data Pipeline/Data/GridSearch.npy'
evaluationResultsFile = './Gene Data Pipeline/Data/EvaluationResults.npy'

class Pipeline:
    def __init__(self):
        if not path.exists("./Gene Data Pipeline/"):
            sys.exit('Please run the file from the root folder.')

        # The purpose of this function is to speed up debugging by not having to read the data set each run
        if (path.exists(rawDataFile)==False or path.exists(labelsFile)==False or path.exists(labelsNameFile)==False):
            print("Reading data from csv file...", end='\r')
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
            print("Reading data from csv file... Done")
        else:
            print("Reading data from binary file...", end='\r')
            self.rawData = np.load(rawDataFile)
            self.labels = np.load(labelsFile)
            self.labelNames = np.load(labelsNameFile, allow_pickle=True)
            print("Reading data from binary file... Done")

    def splitData(self):
        # Shuffle the dataset using a seed
        testSetFactor = 0.1        
        p = np.random.RandomState(0).permutation(len(self.labels))
        rawDataX = self.rawData[p]
        self.dataY = self.labels[p]
        testSetCount = round(len(rawDataX)*testSetFactor)
        self.rawTrainX = rawDataX[0:len(rawDataX)-testSetCount]
        self.trainY = self.dataY[0:len(self.dataY)-testSetCount]
        self.rawTestX = rawDataX[len(rawDataX)-testSetCount:len(rawDataX)]
        self.testY = self.dataY[len(self.dataY)-testSetCount:len(self.dataY)]

    def preProcess(self):
        print("Pre-processing data...", end='\r')
        # Augment the training data
        samplesPerClass = [0]*len(self.trainY)
        for label in self.trainY:
            samplesPerClass[label] +=1
        samplesPerClass = np.max(samplesPerClass)
        oversample = SMOTE(sampling_strategy = {0:samplesPerClass, 1:samplesPerClass, 2:samplesPerClass, 3:samplesPerClass, 4:samplesPerClass}, random_state=27)
        augmentedTrainX, self.augmentedTrainY = oversample.fit_resample(self.rawTrainX, self.trainY)

        # Normalize the whole data set for clustering
        self.dataSetX = (self.rawData-np.mean(self.rawData))/np.std(self.rawData)
        
        # Normalize the original train data
        trainingMean = np.mean(self.rawTrainX, axis=0) 
        centeredTrainingData = self.rawTrainX - trainingMean
        trainingSD = np.std(centeredTrainingData, axis=0)
        zeroColumns = trainingSD == 0
        nonZeroSD = np.delete(trainingSD,zeroColumns,axis=0)
        self.trainX = np.delete(centeredTrainingData,zeroColumns,axis=1)/nonZeroSD

        # Normalize original test set with the normalization parameters from the original train set
        centeredTestingData = self.rawTestX - trainingMean
        self.testX = np.delete(centeredTestingData,zeroColumns,axis=1)/nonZeroSD

        # Normalize the augmented train data
        augmentedMean = np.mean(augmentedTrainX, axis=0) 
        augmentedCenteredData = augmentedTrainX - augmentedMean
        augmentedSD = np.std(augmentedCenteredData, axis=0)
        augmentedZeroColumns = augmentedSD == 0
        augmentedNonZeroSD = np.delete(augmentedSD,augmentedZeroColumns,axis=0)
        self.augmentedTrainX = np.delete(augmentedCenteredData,augmentedZeroColumns,axis=1)/augmentedNonZeroSD

        # Normalize augmented test set with the normalization parameters from the augmented train set
        augmentedCenteredTestingData = self.rawTestX - augmentedMean
        self.augmentedTestX = np.delete(augmentedCenteredTestingData,augmentedZeroColumns,axis=1)/augmentedNonZeroSD
        print("Pre-processing data... Done")
      
    def featureExtraction(self, numberOfComponents):
        '''
        Maps the data to the number of components.

        Returns the reduced sets originalTrainX, augmentedTrainX, dataSetX
        '''
        self.pca = PCA(numberOfComponents)
        originalTrainX = self.pca.fit_transform(self.trainX)

        self.pcaAugmented = PCA(numberOfComponents)
        augmentedTrainX = self.pcaAugmented.fit_transform(self.augmentedTrainX)

        dataSetPca = PCA(numberOfComponents)
        dataSetX = dataSetPca.fit_transform(self.dataSetX)
        return originalTrainX, augmentedTrainX, dataSetX

    def gridSearch(self, pcaComponentSearch):
        print("Grid search...", end='\r')
        # If a grid search already is performed, load the file to only search new components
        if path.exists(GridSearchResultsFile):
            with open (GridSearchResultsFile, 'rb') as fp:
                gridSearchResults = np.array(pickle.load(fp))
                # check which parameters already exist and append new parameters
            searchRange = []
            for parameter in pcaComponentSearch:
                if (parameter in(gridSearchResults[:,0]))==False:
                    searchRange.append(parameter)
        else:
            gridSearchResults=np.array([])
            searchRange = pcaComponentSearch
        
        if len(searchRange)==0:
            print("Grid search... Done")
            return
        
        results = []
        run = 1
        for components in searchRange:           
            result = [components]
            
            if components == len(self.trainX[0]):
                input_data=self.trainX
                dataSetX = self.dataSetX
                targets = self.trainY
            else:
                input_data,_,dataSetX = self.featureExtraction(components)
                targets = self.trainY

            # Knn
            print("PCA search " + str(run) + "/"+str(len(searchRange))+": KNN")
            knn_model = KNeighborsClassifier()
            knn_parameters = {'n_neighbors':[1, 3, 5, 7, 9, 11, 13, 15],'p':[1,2],'weights':('uniform', 'distance')} 
            knn_search = GridSearchCV(knn_model, knn_parameters, cv = LeaveOneOut(), verbose=0)            
            knn_search.fit(input_data, targets)
            result.append(knn_search.best_score_)
            result.append(knn_search.best_params_)
            
            # Logistic regression
            print("PCA search " + str(run) + "/"+str(len(searchRange))+": lr")
            lr_model = LogisticRegression(max_iter=10000, class_weight='balanced')  
            lr_parameters = {'penalty':('l2', 'none')}
            lr_search = GridSearchCV(lr_model, lr_parameters, cv = LeaveOneOut(), verbose=0)
            lr_search.fit(input_data, targets)
            result.append(lr_search.best_score_)
            result.append(lr_search.best_params_)

            # Bayes
            print("PCA search " + str(run) + "/"+str(len(searchRange))+": Naive Bayes")
            bayesModel = GaussianNB()
            bayesParameters = {}
            bayesSearch = GridSearchCV(bayesModel, bayesParameters, cv = LeaveOneOut(), verbose=0)
            bayesSearch.fit(input_data, targets)
            result.append(bayesSearch.best_score_)
            result.append(bayesSearch.best_params_)

            # Set the minimum and maximum numbers of clusters to check
            print("PCA search " + str(run) + "/"+str(len(searchRange))+": FCM")
            clusters_to_check = np.arange(2, 30 + 1, dtype = np.int32)
            SC_original_data = []
            for c in clusters_to_check:
                clm = FCM(n_clusters = c)
                clm.fit(dataSetX)
                labels = clm.predict(dataSetX)
                silhouette_coefficient_original = silhouette_score(dataSetX, labels)
                SC_original_data.append(silhouette_coefficient_original)
            bestIndex = np.argmax(SC_original_data)
            bestScore = SC_original_data[bestIndex]
            bestParameters = {'clusters': clusters_to_check[bestIndex]}
            result.append(bestScore)
            result.append(bestParameters)
            
            results.append(result)
            run +=1

        newResults = results

        if len(gridSearchResults) == 0:
            gridSearchResults = newResults
        else:
            gridSearchResults = np.append(gridSearchResults,newResults, axis=0)
        
        with open(GridSearchResultsFile, 'wb') as fp:
            pickle.dump(gridSearchResults, fp)
        print("Grid search... Done")
    
    def evaluateModels(self, bestKnnPca,bestLrPca,bestBayesPca, bestFcMeansPca):
        # Open de search results to load the optimal model parameters
        with open (GridSearchResultsFile, 'rb') as fp:
            searchResults = np.array(pickle.load(fp))

        # Evaluate models on the reduced data
        print("Evaluating models on the reduced data...", end='\r')

        knnValidationAcc = searchResults[searchResults[:,0]==bestKnnPca][0][1] 
        bestKnnSettings = searchResults[searchResults[:,0]==bestKnnPca][0][2]
        knnPcaResults = ["knn", bestKnnPca, bestKnnSettings, knnValidationAcc]
        knn_model = KNeighborsClassifier(n_neighbors = bestKnnSettings['n_neighbors'], weights = bestKnnSettings['weights'], p=bestKnnSettings['p'])
        knnPcaResults.extend(self.evaluateClassification(knn_model, bestKnnPca))
    
        lrValidationAcc = searchResults[searchResults[:,0]==bestLrPca][0][3] 
        bestLrSettings = searchResults[searchResults[:,0]==bestLrPca][0][4]
        lrPcaResults = ["lr", bestLrPca, bestLrSettings, lrValidationAcc]
        lr_model = LogisticRegression(penalty=bestLrSettings['penalty'], max_iter=10000, class_weight='balanced')  
        lrPcaResults.extend(self.evaluateClassification(lr_model, bestLrPca))
        
        bayesValidationAcc = searchResults[searchResults[:,0]==bestBayesPca][0][5] 
        bestBayesSettings = searchResults[searchResults[:,0]==bestBayesPca][0][6]
        BayesPcaResults = ["Bayes",bestBayesPca,bestBayesSettings, bayesValidationAcc]
        bayesModel = GaussianNB()
        BayesPcaResults.extend(self.evaluateClassification(bayesModel, bestBayesPca))

        bestFcMeanSettings = searchResults[searchResults[:,0]==bestFcMeansPca][0][8]
        FcPcaResults = ["FC-means",bestFcMeansPca,bestFcMeanSettings, 0, searchResults[searchResults[:,0]==bestFcMeansPca][0][7], 0,0,0]
        print("Evaluating models on the reduced data... Done")

        # Evaluate all models on the original data
        print("Evaluating models on the unreduced data...", end='\r')
        lastRow = searchResults[len(searchResults[:,0])-1]
        
        knnValidationAcc = lastRow[1]
        bestKnnSettings = lastRow[2]
        knnResults = ["knn", len(self.trainX[0]), bestKnnSettings,knnValidationAcc]
        knn_model = KNeighborsClassifier(n_neighbors = bestKnnSettings['n_neighbors'], weights = bestKnnSettings['weights'], p=bestKnnSettings['p'])
        knnResults.extend(self.evaluateClassification(knn_model))

        lrValidationAcc = lastRow[3]
        bestLrSettings = lastRow[4]
        lrResults = ["lr", len(self.trainX[0]), bestLrSettings,lrValidationAcc]
        lr_model = LogisticRegression(penalty=bestLrSettings['penalty'], max_iter=10000, class_weight='balanced')  
        lrResults.extend(self.evaluateClassification(lr_model))

        bayesValidationAcc = lastRow[5]
        bestBayesSettings = lastRow[6]
        BayesResults = ["Bayes",len(self.trainX[0]),bestBayesSettings,bayesValidationAcc]
        bayesModel = GaussianNB()
        BayesResults.extend(self.evaluateClassification(bayesModel))

        bestFcMeanSettings = lastRow[8]
        FcResults = ["FCM",len(self.trainX[0]),bestFcMeanSettings,0,lastRow[7], 0,0,0]
        
        print("Evaluating models on the unreduced data... Done")

        # ModelName, Components, best settings, standardAccuracy, standardPredictions, augmentationAccuracy, augmentationPredictions
        evaluationResults = [knnPcaResults, lrPcaResults, BayesPcaResults, FcPcaResults, knnResults, lrResults, BayesResults, FcResults]
        with open(evaluationResultsFile, 'wb') as fp:
            pickle.dump(evaluationResults, fp)
        return evaluationResults
             
    def evaluateClassification(self, model, pcaComponents=0):
        '''
        Trains the classification model on the training set, and tests it on the test set.
        
        Returns [standardAccuracy, standardPredictions, augmentationAccuracy, augmentationPredictions]
        '''
        # Evaluate the model on the test data
        if pcaComponents>0:
            trainX, augmentedTrainX, _ = self.featureExtraction(pcaComponents)
            testX = self.pca.transform(self.testX)
            augmentedTestX = self.pcaAugmented.transform(self.augmentedTestX)
        else:
            trainX = self.trainX
            testX = self.testX
            augmentedTrainX = self.augmentedTrainX
            augmentedTestX = self.augmentedTestX 
        
        # Copy the model such that there is an untrained model for augmentation
        modelAugmented = clone(model)
        
        # Original data set
        model.fit(trainX, self.trainY)
        orPredictions = model.predict(testX)
        orPredictions = np.vstack((self.testY, orPredictions)).T
        CorrectPredicted = orPredictions[:,0] == orPredictions[:,1]
        originalAccuracy = np.sum(CorrectPredicted)/len(orPredictions)
    
        # Augmented data set
        modelAugmented.fit(augmentedTrainX, self.augmentedTrainY)
        augPredictions = modelAugmented.predict(augmentedTestX)
        augPredictions = np.vstack((self.testY, augPredictions)).T
        CorrectPredicted = augPredictions[:,0] == augPredictions[:,1]
        augAccuracy = np.sum(CorrectPredicted)/len(augPredictions)
        return [originalAccuracy,orPredictions,augAccuracy,augPredictions]

if __name__=="__main__":
    pipeline = Pipeline()
    pipeline.splitData()
    pipeline.preProcess()

    # Specify the array for which pca components the grid search must be conducted
    pcaComponentSearch = [*range(1, 40, 1)]
    pcaComponentSearch.extend([*range(40, 171, 10)])
    pcaComponentSearch.extend([len(pipeline.trainX[0])])
    pipeline.gridSearch(pcaComponentSearch)

    bestKnnPca = 13
    bestLrPca = 12
    bestBayesPca = 9
    bestFcMeansPca = 1
    evaluationResults = pipeline.evaluateModels(bestKnnPca,bestLrPca,bestBayesPca, bestFcMeansPca)

    # Mask the predictions in the array to plot the results table.
    simpleResults = np.array(evaluationResults,dtype=object)[:,[0,1,2,3,4,6]]
    df = pd.DataFrame(simpleResults, columns = ['Model','Dimensions','Best settings', 'validation perf.' ,'original perf', 'augmentation perf.'])
    print()
    print(df)