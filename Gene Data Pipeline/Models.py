import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier


class ModelLR:
    def __init__(self):
        self.regressionModel = LogisticRegression(random_state=16)
    
    def newModel(self):
        pass

    def train(self, trainX, trainY, classWeights):
        self.regressionModel = LogisticRegression(random_state=16, max_iter=10000, class_weight='balanced')
        self.regressionModel.fit(trainX, trainY)

    def test(self, testX, testY):
        predictY = self.regressionModel.predict(testX)
        accuracy = testY==predictY
        return accuracy

class ModelKNN:
    def __init__(self, parameterSet):
        k = parameterSet['k']
        p = parameterSet['p']
        k = parameterSet['k']
        self.knn_model = KNeighborsClassifier(n_neighbors=k)
    
    def newModel(self):
        pass

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
        self.labels = None
        self.components = 0
    
    def newModel(self):
        self.components = 0

    def train(self, trainX, trainY,classWeights):
        self.gmm = None
        if self.components == 0:
            AIC = []
            gmms = []
            componentsRange = range(5,16,1)
            for components in componentsRange:
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
                newGmm.fit(trainX) 
                AIC.append(newGmm.aic(trainX))
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
            self.components = self.gmm.n_components
            print("Optimal number of Gaussian components is : " + str(self.components))
        else:
            np.random.seed(42)
            self.gmm = GaussianMixture(n_components=self.components)
            self.gmm.fit(trainX) 

        # Match the labels of the MoG with the real labels. 
        labels = np.zeros((self.components, 5))
        for i in range(len(trainY)):
            realLabel = trainY[i]
            image = trainX[i]
            image = image.reshape(1,-1)
            classPredictions = int(self.gmm.predict(image))
            labels[classPredictions][realLabel] += 1
        self.labels = [0] * self.components
        good = 0
        for index in range(self.components):
            mostCount = np.argmax(labels[index])
            good += labels[index][mostCount]
            self.labels[index] = mostCount
        
        # calculate the accuracy of the trained model on the training data
        accuracy = good/len(trainY)

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