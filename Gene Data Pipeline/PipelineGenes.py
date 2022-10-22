import pickle
import numpy as np
from numpy import genfromtxt
from os import path
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut

# File locations
dataFileName = './Data/Genes/data.csv'
labelsFileName = './Data/Genes/labels.csv'

# Storage of data to speed up debugging
rawDataFile = './Gene Data Pipeline/Data/rawData.npy'
labelsFile = './Gene Data Pipeline/Data/labels.npy'
labelsNameFile = './Gene Data Pipeline/Data/labelNames.npy'
GridSearchClassifiersFile = './Gene Data Pipeline/Data/ClassifiersGridSearch.npy'
#roughGridSearchFile = './Gene Data Pipeline/Data/roughGrid.npy'
#fineGridSearchFile = './Gene Data Pipeline/Data/fineGrid.npy'
#finerGridSearchFile= './Gene Data Pipeline/Data/finerGrid.npy'

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
        return optimalComponent
      
    def featureExtraction(self, numberOfComponents):
        self.pca = PCA(numberOfComponents)
        reducedDimensionsData = self.pca.fit_transform(self.trainX)
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

    def gridSearch(self, componentsList):
        results = []
        for components in componentsList:
            result = [components]
            
            # Set up validation data
            input_data = pipeline.featureExtraction(components)
            targets = pipeline.trainY

            # Knn
            print("Starting grid search with "+str(components)+" PCA components for knn.")
            knn_model = KNeighborsClassifier()
            knn_parameters = {'n_neighbors':[1, 3, 5, 7, 9, 11, 13, 15],'p':[1,2],'weights':('uniform', 'distance')} 
            knn_search = GridSearchCV(knn_model, knn_parameters, cv = LeaveOneOut(), verbose=1)            
            #knn_parameters = {}
            # knn_search = GridSearchCV(knn_model, knn_parameters, verbose=2)
            knn_search.fit(input_data, targets)
            #print(f"With a validation accuracy of {knn_search.best_score_}%, the best combination of hyperparameter settings for KNN is:")
            result.append(knn_search.best_score_)
            result.append(knn_search.best_params_)
            
            # # Logistic regression
            print("Starting grid search with "+str(components)+" PCA components for lr.")
            lr_model = LogisticRegression(max_iter=10000, class_weight='balanced')  
            lr_parameters = {'penalty':('l2', 'none')}
            lr_search = GridSearchCV(lr_model, lr_parameters, cv = LeaveOneOut(), verbose=1)
            # lr_parameters = {}
            # lr_search = GridSearchCV(lr_model, lr_parameters, verbose=2)
            lr_search.fit(input_data, targets)
            #print(f"With a validation accuracy of {lr_search.best_score_}%, the best combination of hyperparameter settings for Logistic Regression is:")
            result.append(lr_search.best_score_)
            result.append(lr_search.best_params_)
            
            # Collect results
            results.append(result.copy())
        return results

    def evaluate(self, model, pcaComponents):
        # Evaluate the model on the test data
        trainX = pipeline.featureExtraction(pcaComponents)
        testX = pipeline.pca.transform(pipeline.testX)
        
        model.fit(trainX, pipeline.trainY)
        predictions = model.predict(testX)
        result = np.vstack((pipeline.testY, predictions)).T
        return result


if __name__=="__main__":
    pipeline = Pipeline()
    pipeline.splitData()
    pipeline.preProcess()
    estimatedComponents = pipeline.pcaSearch() 
    
    # Specify the array for which pca components the grid search must be conducted
    pcaComponentSearch = [*range(1, 40, 1)]
    pcaComponentSearch.extend([*range(40, estimatedComponents+1, 10)])

    # If a grid search already is performed, load the file to only search new components
    if path.exists(GridSearchClassifiersFile):
        with open (GridSearchClassifiersFile, 'rb') as fp:
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
    
    if len(searchRange)>0:
        newResults = pipeline.gridSearch(searchRange)
        if len(gridSearchResults) == 0:
            gridSearchResults = newResults
        else:
            gridSearchResults = np.append(gridSearchResults,newResults, axis=0)
        
        with open(GridSearchClassifiersFile, 'wb') as fp:
            pickle.dump(gridSearchResults, fp)

    bestKnnPca = 8
    bestLrPca = 12

    # Open de search results to load the optimal model parameters
    with open (GridSearchClassifiersFile, 'rb') as fp:
        searchResults = np.array(pickle.load(fp))

    # Evaluate all models on the test data
    print("The class label names are "+str(pipeline.labelNames))
    bestKnnSettings = searchResults[searchResults[:,0]==bestKnnPca][0][2]
    knn_model = KNeighborsClassifier(n_neighbors = bestKnnSettings['n_neighbors'], weights = bestKnnSettings['weights'], p=bestKnnSettings['p'])
    knnPredictions = pipeline.evaluate(knn_model, bestKnnPca)
    knnCorrectPredicted = knnPredictions[:,0] == knnPredictions[:,1]
    knnDifferentIndices = np.where(knnCorrectPredicted==False)
    knnAccuracy = np.sum(knnCorrectPredicted)/len(knnPredictions)
    print("KNN classifier test results:")
    print("  The best model settings for " +str(bestKnnPca)+" PCA components are " + str(bestKnnSettings))
    print("  The test accuracy is: " + str(knnAccuracy))
    print("  Indices " + str(knnDifferentIndices)+ " of the test are [labeled, prediced] as: " + str(knnPredictions[knnDifferentIndices]))

    bestLrSettings = searchResults[searchResults[:,0]==bestLrPca][0][4]
    lr_model = LogisticRegression(penalty=bestLrSettings['penalty'], max_iter=10000, class_weight='balanced')  
    lrPredictions = pipeline.evaluate(lr_model, bestLrPca)
    lrCorrectPredicted = lrPredictions[:,0] == lrPredictions[:,1]
    lrDifferentIndices = np.where(lrCorrectPredicted==False)
    lrAccuracy = np.sum(lrCorrectPredicted)/len(lrPredictions)
    print("LR classifier test results:")
    print("  The best model settings for " +str(bestLrPca)+" PCA components are " + str(bestLrSettings))
    print("  The test accuracy is: " + str(lrAccuracy))
    print("  Indices " + str(lrDifferentIndices)+ " of the test are [labeled, prediced] as: " + str(lrPredictions[lrDifferentIndices]))