import pickle
import numpy as np
from numpy import genfromtxt
from os import path
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut
import time
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE
from fcmeans import FCM
from sklearn.metrics import silhouette_score
from sklearn.base import clone
import pandas as pd

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
        # The purpose of this function is to speed up debugging by not having to read the data set each run
        if (path.exists(rawDataFile)==False or path.exists(labelsFile)==False or path.exists(labelsNameFile)==False): 
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
        return optimalComponent
      
    def featureExtraction(self, numberOfComponents):
        self.pca = PCA(numberOfComponents)
        self.pcaAugmented = PCA(numberOfComponents)
        dataSetPca = PCA(numberOfComponents)
        reducedDimensionsData = self.pca.fit_transform(self.trainX)
        AugmentedReducedDimensionsData = self.pcaAugmented.fit_transform(self.augmentedTrainX)
        dataSetX = dataSetPca.fit_transform(self.dataSetX)
        return reducedDimensionsData, AugmentedReducedDimensionsData, dataSetX

    def gridSearch(self, pcaComponentSearch):
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
        # with open (roughGridSearchFile, 'rb') as fp:
        #         tempResults = pickle.load(fp)
        # newResults = tempResults
        
        if len(searchRange)==0:
            return

        results = []
        run = 1
        for components in searchRange:
            print("Starting search " + str(run) + "/"+str(len(searchRange)))
            run +=1
            start_time = time.time()
            result = [components]
            
            if components == len(self.trainX[0]):
                input_data=pipeline.trainX
                dataSetX = self.dataSetX
                targets = pipeline.trainY
            else:
                input_data,_,dataSetX = pipeline.featureExtraction(components)
                targets = pipeline.trainY

            # Knn
            print("Starting grid search with "+str(components)+" components for knn.")
            knn_model = KNeighborsClassifier()
            knn_parameters = {'n_neighbors':[1, 3, 5, 7, 9, 11, 13, 15],'p':[1,2],'weights':('uniform', 'distance')} 
            knn_search = GridSearchCV(knn_model, knn_parameters, cv = LeaveOneOut(), verbose=2)            
            knn_search.fit(input_data, targets)
            result.append(knn_search.best_score_)
            result.append(knn_search.best_params_)
            
            # Logistic regression
            print("Starting grid search with "+str(components)+" components for lr.")
            lr_model = LogisticRegression(max_iter=10000, class_weight='balanced')  
            lr_parameters = {'penalty':('l2', 'none')}
            lr_search = GridSearchCV(lr_model, lr_parameters, cv = LeaveOneOut(), verbose=2)
            lr_search.fit(input_data, targets)
            result.append(lr_search.best_score_)
            result.append(lr_search.best_params_)

            # Bayes
            print("Starting grid search with "+str(components)+" components for Naive Bayes")
            bayesModel = GaussianNB()
            bayesParameters = {}
            bayesSearch = GridSearchCV(bayesModel, bayesParameters, cv = LeaveOneOut(), verbose=2)
            bayesSearch.fit(input_data, targets)
            result.append(bayesSearch.best_score_)
            result.append(bayesSearch.best_params_)

            # Set the minimum and maximum numbers of clusters to check
            minimum_number_of_clusters = 2
            maximum_number_of_clusters = 30
            clusters_to_check = np.arange(minimum_number_of_clusters, maximum_number_of_clusters + 1, dtype = np.int32)
            # For each number of clusters to check, fit a fuzzy c-means model 
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
            # Collect results
            print("The search took %s seconds." % round((time.time() - start_time),2))
            results.append(result)

        newResults = results

        if len(gridSearchResults) == 0:
            gridSearchResults = newResults
        else:
            gridSearchResults = np.append(gridSearchResults,newResults, axis=0)
        
        with open(GridSearchResultsFile, 'wb') as fp:
            pickle.dump(gridSearchResults, fp)
    
    def evaluateModels(self, bestKnnPca,bestLrPca,bestBayesPca, bestFcMeansPca):
        # Open de search results to load the optimal model parameters
        with open (GridSearchResultsFile, 'rb') as fp:
            searchResults = np.array(pickle.load(fp))

        # Evaluate models on the best-reduced data
        print("Evaluating models on best-reduced data set...")

        knnValidationAcc = searchResults[searchResults[:,0]==bestKnnPca][0][1] 
        bestKnnSettings = searchResults[searchResults[:,0]==bestKnnPca][0][2]
        knnPcaResults = ["knn", bestKnnPca, bestKnnSettings, knnValidationAcc]
        knn_model = KNeighborsClassifier(n_neighbors = bestKnnSettings['n_neighbors'], weights = bestKnnSettings['weights'], p=bestKnnSettings['p'])
        knnPcaResults.extend(self.evaluateClassificationBestReduced(knn_model, bestKnnPca))
    
        lrValidationAcc = searchResults[searchResults[:,0]==bestLrPca][0][3] 
        bestLrSettings = searchResults[searchResults[:,0]==bestLrPca][0][4]
        lrPcaResults = ["lr", bestLrPca, bestLrSettings, lrValidationAcc]
        lr_model = LogisticRegression(penalty=bestLrSettings['penalty'], max_iter=10000, class_weight='balanced')  
        lrPcaResults.extend(self.evaluateClassificationBestReduced(lr_model, bestLrPca))
        
        bayesValidationAcc = searchResults[searchResults[:,0]==bestBayesPca][0][5] 
        bestBayesSettings = searchResults[searchResults[:,0]==bestBayesPca][0][6]
        BayesPcaResults = ["Bayes",bestBayesPca,bestBayesSettings, bayesValidationAcc]
        bayesModel = GaussianNB()
        BayesPcaResults.extend(self.evaluateClassificationBestReduced(bayesModel, bestBayesPca))

        bestFcMeanSettings = searchResults[searchResults[:,0]==bestFcMeansPca][0][8]
        FcPcaResults = ["FC-means",bestFcMeansPca,bestFcMeanSettings, 0, searchResults[searchResults[:,0]==bestFcMeansPca][0][7], 0,0,0]

        # Evaluate all models on the original data
        print("Evaluating models on the original data set...")
        lastRow = searchResults[len(searchResults[:,0])-1]
        
        knnValidationAcc = lastRow[1]
        bestKnnSettings = lastRow[2]
        knnResults = ["knn", len(self.trainX[0]), bestKnnSettings,knnValidationAcc]
        knn_model = KNeighborsClassifier(n_neighbors = bestKnnSettings['n_neighbors'], weights = bestKnnSettings['weights'], p=bestKnnSettings['p'])
        knnResults.extend(self.evaluateClassificationOriginal(knn_model))

        lrValidationAcc = lastRow[3]
        bestLrSettings = lastRow[4]
        lrResults = ["lr", len(self.trainX[0]), bestLrSettings,lrValidationAcc]
        lr_model = LogisticRegression(penalty=bestLrSettings['penalty'], max_iter=10000, class_weight='balanced')  
        lrResults.extend(self.evaluateClassificationOriginal(lr_model))

        bayesValidationAcc = lastRow[5]
        bestBayesSettings = lastRow[6]
        BayesResults = ["Bayes",len(self.trainX[0]),bestBayesSettings,bayesValidationAcc]
        bayesModel = GaussianNB()
        BayesResults.extend(self.evaluateClassificationOriginal(bayesModel))

        bestFcMeanSettings = lastRow[8]
        FcResults = ["FC-means",len(self.trainX[0]),bestFcMeanSettings,0,lastRow[7], 0,0,0]

        # ModelName, Components, best settings, standardAccuracy, standardPredictions, augmentationAccuracy, augmentationPredictions
        evaluationResults = [knnPcaResults, lrPcaResults, BayesPcaResults, FcPcaResults, knnResults, lrResults, BayesResults, FcResults]
        with open(evaluationResultsFile, 'wb') as fp:
            pickle.dump(evaluationResults, fp)
        return evaluationResults
             
    def evaluateClassificationBestReduced(self, model, pcaComponents):
        # Evaluate the model on the test data
        trainX, augmentedTrainX, _ = pipeline.featureExtraction(pcaComponents)
        testX = pipeline.pca.transform(pipeline.testX)
        augmentedTestX = pipeline.pcaAugmented.transform(pipeline.augmentedTestX)

        # [standardAccuracy, standardPredictions, augmentationAccuracy, augmentationPredictions]
        result = []
        
        # Copy the model such that there is an untrained model for augmentation
        modelAugmented = clone(model)
        
        # Original data set
        model.fit(trainX, pipeline.trainY)
        predictions = model.predict(testX)
        predictions = np.vstack((pipeline.testY, predictions)).T
        
        CorrectPredicted = predictions[:,0] == predictions[:,1]
        Accuracy = np.sum(CorrectPredicted)/len(predictions)
        result.append(Accuracy)
        result.append(predictions)
    
        # Augmented data set
        modelAugmented.fit(augmentedTrainX, pipeline.augmentedTrainY)
        predictions = modelAugmented.predict(augmentedTestX)
        predictions = np.vstack((pipeline.testY, predictions)).T
        
        CorrectPredicted = predictions[:,0] == predictions[:,1]
        Accuracy = np.sum(CorrectPredicted)/len(predictions)
        result.append(Accuracy)
        result.append(predictions)
        return result

    def evaluateClassificationOriginal(self, model):
        result = []
        # Original data set
        modelAugmented = clone(model)
        model.fit(pipeline.trainX, pipeline.trainY)
        predictions = model.predict(pipeline.testX)
        predictions = np.vstack((pipeline.testY, predictions)).T
        
        CorrectPredicted = predictions[:,0] == predictions[:,1]
        Accuracy = np.sum(CorrectPredicted)/len(predictions)
        result.append(Accuracy)
        result.append(predictions)

        modelAugmented.fit(self.augmentedTrainX, pipeline.augmentedTrainY)
        predictions = modelAugmented.predict(self.augmentedTestX)
        predictions = np.vstack((pipeline.testY, predictions)).T
        
        CorrectPredicted = predictions[:,0] == predictions[:,1]
        Accuracy = np.sum(CorrectPredicted)/len(predictions)

        result.append(Accuracy)
        result.append(predictions)
        return result

if __name__=="__main__":
    pipeline = Pipeline()
    pipeline.splitData()
    pipeline.preProcess()
    estimatedComponents = pipeline.pcaSearch() 

    # Specify the array for which pca components the grid search must be conducted
    pcaComponentSearch = [*range(1, 40, 1)]
    pcaComponentSearch.extend([*range(40, estimatedComponents+1, 10)])
    pcaComponentSearch.extend([len(pipeline.trainX[0])])
    pipeline.gridSearch(pcaComponentSearch)

    bestKnnPca = 13
    bestLrPca = 12
    bestBayesPca = 9
    bestFcMeansPca = 1

    print()
    print("Start of evaluation of the models")
    print("The class label names are "+str(pipeline.labelNames))
    print()
    evaluationResults = pipeline.evaluateModels(bestKnnPca,bestLrPca,bestBayesPca, bestFcMeansPca)

    # Mask the predictions in the array to plot the results table.
    simpleResults = np.array(evaluationResults)[:,[True,True,True,True, True,False,True,False]]
    df = pd.DataFrame(simpleResults, columns = ['Model','Dimensions','Best settings', 'val. performance' ,'standard perf.', 'augmentation perf.'])
    print()
    print(df)