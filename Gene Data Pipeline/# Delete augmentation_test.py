# Generate and plot a synthetic imbalanced classification dataset
from collections import Counter
from sklearn.datasets import make_classification
from matplotlib import pyplot
from numpy import where
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

# Read in raw data
rawDataFile = './Gene Data Pipeline/Data/rawData.npy'
rawData = np.load(rawDataFile)
labelsFile = './Gene Data Pipeline/Data/labels.npy'
labels = np.load(labelsFile)

# Split the data into training and test
testSetFactor = 0.1
p = np.random.RandomState(0).permutation(len(labels))
rawDataX = rawData[p]
dataY = labels[p]
testSetCount = round(len(rawDataX)*testSetFactor)
rawTrainX = rawDataX[0:len(rawDataX)-testSetCount]
trainY = dataY[0:len(dataY)-testSetCount]
rawTestX = rawDataX[len(rawDataX)-testSetCount:len(rawDataX)]
testY = dataY[len(dataY)-testSetCount:len(dataY)]

# Augment the training data
oversample = SMOTE(sampling_strategy = {0:273, 1:273, 2:273, 3:273, 4:273})
trainX, trainY = oversample.fit_resample(rawTrainX, trainY)


# Preprocess the augmented data for PCA
inputMean = np.mean(trainX, axis = 0)
centeredTrainingData = trainX - inputMean
inputSD = np.std(centeredTrainingData, axis = 0)
zeroColumns = inputSD == 0
nonZeroSD = np.delete(inputSD, zeroColumns, axis = 0)
preprocessedTrainingData = np.delete(centeredTrainingData, zeroColumns, axis = 1)/nonZeroSD

centeredTestData = rawTestX - inputMean
preprocessedTestData = np.delete(centeredTestData, zeroColumns, axis = 1)/nonZeroSD


# Train KNN model with earlier found best settings on reduced augmented data
knn_PCA = PCA(12)
knn_reducedTrainingData = knn_PCA.fit_transform(preprocessedTrainingData)
knn_model = KNeighborsClassifier(n_neighbors=9, p=2, weights='uniform')
knn_model.fit(knn_reducedTrainingData, trainY)

# Test the trained KNN model on reduced test data
knn_reducedTestData = knn_PCA.transform(preprocessedTestData)
print(f"KNN's score on augmented data: {knn_model.score(knn_reducedTestData, testY)}")


# Train Logistic Regression model with earlier found best settings on reduced augmented data
lr_PCA = PCA(11)
lr_reducedTrainingData = lr_PCA.fit_transform(preprocessedTrainingData)
lr_model = LogisticRegression(penalty = 'l2', max_iter = 500)
lr_model.fit(lr_reducedTrainingData, trainY)

# Test the trained LR model on reduced test data
lr_reducedTestData = lr_PCA.transform(preprocessedTestData)
print(f"LR's score on augmented data: {lr_model.score(lr_reducedTestData, testY)}")
