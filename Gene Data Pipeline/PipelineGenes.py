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

# File locations
dataFileName = './Data/Genes/data.csv'
labelsFileName = './Data/Genes/labels.csv'

# Storage of data to speed up debugging
rawDataFile = './Gene Data Pipeline/Data/rawData.npy'
labelsFile = './Gene Data Pipeline/Data/labels.npy'
labelsNameFile = './Gene Data Pipeline/Data/labelNames.npy'
GridSearchFile = './Gene Data Pipeline/Data/GridSearch.npy'

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

        # Apply same normalization on the test data
        centeredTestingData = self.rawTestX - trainingMean
        self.testX = np.delete(centeredTestingData,zeroColumns,axis=1)/nonZeroSD

        # Augment the training data
        samplerPerClass = 273
        oversample = SMOTE(sampling_strategy = {0:samplerPerClass, 1:samplerPerClass, 2:samplerPerClass, 3:samplerPerClass, 4:samplerPerClass})
        augmentedTrainX, self.augmentedTrainY = oversample.fit_resample(self.rawTrainX, self.trainY)
        augmentedMean = np.mean(augmentedTrainX, axis=0) 
        augmentedCenteredData = augmentedTrainX - augmentedMean
        augmentedSD = np.std(augmentedCenteredData, axis=0)
        augmentedZeroColumns = augmentedSD == 0
        augmentedNonZeroSD = np.delete(augmentedSD,augmentedZeroColumns,axis=0)
        self.augmentedTrainX = np.delete(augmentedCenteredData,augmentedZeroColumns,axis=1)/augmentedNonZeroSD

        # Apply same normalization on the test data
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
        reducedDimensionsData = self.pca.fit_transform(self.trainX)
        AugmentedReducedDimensionsData = self.pcaAugmented.fit_transform(self.augmentedTrainX)
        return reducedDimensionsData, AugmentedReducedDimensionsData

# Grid search on training data. Both original and augmented 
    def gridSearch(self, pcaComponentSearch):
        # If a grid search already is performed, load the file to only search new components
        if path.exists(GridSearchFile):
            with open (GridSearchFile, 'rb') as fp:
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
                targets = pipeline.trainY
            else:
                input_data,_ = pipeline.featureExtraction(components)
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
                clm.fit(input_data)
                labels = clm.predict(input_data)
                silhouette_coefficient_original = silhouette_score(input_data, labels)
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
        #newResults = self.gridSearchPca(searchRange)
        # Grid search on the original data set
        #result = self.gridSearchOriginal(self)

        if len(gridSearchResults) == 0:
            gridSearchResults = newResults
        else:
            gridSearchResults = np.append(gridSearchResults,newResults, axis=0)
        
        with open(GridSearchFile, 'wb') as fp:
            pickle.dump(gridSearchResults, fp)
    
# Evaluation on the test data
    def evaluateModelsPca(self, bestKnnPca,bestLrPca,bestBayesPca, bestFcMeansPca):
        # Open de search results to load the optimal model parameters
        with open (GridSearchFile, 'rb') as fp:
            searchResults = np.array(pickle.load(fp))

        # Evaluate all models on the test data
        print("The class label names are "+str(self.labelNames))
        
        bestKnnSettings = searchResults[searchResults[:,0]==bestKnnPca][0][2]
        print("KNN classifier test results:")
        print("  The best model settings for " +str(bestKnnPca)+" PCA components are " + str(bestKnnSettings))
        knn_model = KNeighborsClassifier(n_neighbors = bestKnnSettings['n_neighbors'], weights = bestKnnSettings['weights'], p=bestKnnSettings['p'])
        self.evaluatePca(knn_model, bestKnnPca)

        bestLrSettings = searchResults[searchResults[:,0]==bestLrPca][0][4]
        print("LR classifier test results:")
        print("  The best model settings for " +str(bestLrPca)+" PCA components are " + str(bestLrSettings))
        lr_model = LogisticRegression(penalty=bestLrSettings['penalty'], max_iter=10000, class_weight='balanced')  
        self.evaluatePca(lr_model, bestLrPca)

        bestBayesSettings = searchResults[searchResults[:,0]==bestLrPca][0][6]
        print("Bayes classifier test results:")
        print("  The best model settings for " +str(bestBayesPca)+" PCA components are " + str(bestBayesSettings))
        bayesModel = GaussianNB()
        self.evaluatePca(bayesModel, bestBayesPca)

        # bestFcMeanSettings = searchResults[searchResults[:,0]==bestLrPca][0][8]
        # print("FC-means classifier test results:")
        # print("  The best model settings for " +str(bestFcMeansPca)+" PCA components are " + str(bestFcMeanSettings))
        # clm = FCM(n_clusters = bestFcMeanSettings['clusters'])
        # self.evaluatePca(clm, bestFcMeansPca)
       
    def evaluatePca(self, model, pcaComponents):
        # Evaluate the model on the test data
        trainX, augmentedTrainX = pipeline.featureExtraction(pcaComponents)
        testX = pipeline.pca.transform(pipeline.testX)
        augmentedTestX = pipeline.pcaAugmented.transform(pipeline.augmentedTestX)
        


        # Copy the model such that there is an untrained model for augmentation
        modelAugmented = model
        model.fit(trainX, pipeline.trainY)
        modelAugmented.fit(augmentedTrainX, pipeline.augmentedTrainY)
        predictions = model.predict(testX)
        predictions = np.vstack((pipeline.testY, predictions)).T
        
        CorrectPredicted = predictions[:,0] == predictions[:,1]
        DifferentIndices = np.where(CorrectPredicted==False)
        Accuracy = np.sum(CorrectPredicted)/len(predictions)
        print("  PCA on original data set")
        print("    The test accuracy is: " + str(Accuracy))
        print("    Indices " + str(DifferentIndices)+ " of the test are [labeled, prediced] as: " + str(predictions[DifferentIndices]))
    
        predictions = modelAugmented.predict(augmentedTestX)
        predictions = np.vstack((pipeline.testY, predictions)).T
        
        CorrectPredicted = predictions[:,0] == predictions[:,1]
        DifferentIndices = np.where(CorrectPredicted==False)
        Accuracy = np.sum(CorrectPredicted)/len(predictions)
        print("  PCA on augmented data set")
        print("    The test accuracy is: " + str(Accuracy))
        print("    Indices " + str(DifferentIndices)+ " of the test are [labeled, prediced] as: " + str(predictions[DifferentIndices]))

    def evaluateModelsOriginal(self):
        pass

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

    bestKnnPca = 12
    bestLrPca = 11
    bestBayesPca = 8
    bestFcMeansPca = 1

    pipeline.evaluateModelsPca(bestKnnPca,bestLrPca,bestBayesPca, bestFcMeansPca)
    pipeline.evaluateModelsOriginal()