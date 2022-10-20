import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier


class ModelLR:
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

class ModelKNN:
    def __init__(self, k, seed):
        self.knn_model = KNeighborsClassifier(n_neighbors=k)

    def train(self, trainX, trainY, classWeights):
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

class ModelMoG:
    def __init__(self):
        self.gmm = None
        self.labels = None

    def train(self, datasetX, datasetY,classWeights):
        #datasetX = self.trainX
        #datasetY = self.trainY
        
        AIC = []
        gmms = []
        componentsRange = range(5,16,1)
        for components in componentsRange:
            #np.random.seed(42)
            np.random.seed(42)
            newGmm = GaussianMixture(n_components=components)#, random_state=42)
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



        self.gmm = gmms[np.argmin(AIC)]
        self.gmm.n_components
        finalComponents = self.gmm.n_components
        
        # Match the labels of the MoG with the real labels. 
        labels = np.zeros((finalComponents, 5))
        for i in range(len(datasetY)):
            realLabel = datasetY[i]
            image = datasetX[i]
            image = image.reshape(1,-1)
            classPredictions = int(self.gmm.predict(image))
            labels[classPredictions][realLabel] += 1
        self.labels = [0] * finalComponents
        good = 0
        for index in range(finalComponents):
            mostCount = np.argmax(labels[index])
            good += labels[index][mostCount]
            self.labels[index] = mostCount
        
        # calculate the accuracy of the trained model on the training data
        accuracy = good/len(datasetY)

    def test(self, testData, testLabels):
        correct = 0
        for sample, label in zip(testData,testLabels):
            sample = sample.reshape(1,-1)
            predictedComponent = int(self.gmm.predict(sample))
            prediction = self.labels[predictedComponent]
            if prediction == label:
                correct+=1
        accuracy = correct/len(testLabels)
        print(accuracy)
        return accuracy
    
    
