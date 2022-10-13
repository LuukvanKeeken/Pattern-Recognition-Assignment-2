import numpy as np
from numpy import genfromtxt
from os import path
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# File locations
dataFileName = './Data/Genes/data.csv'
labelsFileName = './Data/Genes/labels.csv'

# Storage of data to speed up debugging
rawDataFile = './PreProcessedData/rawData.npy'
rawLabelsFile = './PreProcessedData/rawLabels.npy'
preProcessedDataFile = './PreProcessedData/preProcessedData.npy'
preProcessedLabelsFile = './PreProcessedData/preProcessedLabels.npy'

reProcessRawData = False
reProcessPreprocessedData = False

class Pipeline:
    def __init__(self):
        if (path.exists(rawDataFile)==False or path.exists(rawLabelsFile)==False or reProcessRawData):
            self.labelsDict ={ }
            self.rawData = genfromtxt(dataFileName, skip_header=True, delimiter=',')[:,1:] # shape is 802x20532 

            # Read the labels files, and assign a class number to each label text
            readLabels = np.genfromtxt(labelsFileName, skip_header=True, delimiter=',',dtype=str)[:,1:] # shape is 802x2
            self.rawLabels = []
            dictIndex = 0
            for label in readLabels:
                labelValue = label[0]
                if labelValue in self.labelsDict:
                    encodedLabel = self.labelsDict[labelValue]
                    self.rawLabels.append(encodedLabel)
                else:
                    self.labelsDict[labelValue]=dictIndex
                    dictIndex+=1
            np.save(rawDataFile, self.rawData)
            np.save(rawLabelsFile, self.rawLabels)
        else:
            self.rawData = np.load(rawDataFile)
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

        nanArguments = np.argwhere(np.isnan(self.preProcessedData))


        pca = PCA(n_components=2)
        principalComponents = pca.fit_transform(self.preProcessedData)
        
        fig = plt.figure(figsize = (8,8))
        plt.scatter(principalComponents[:,0],principalComponents[:,1])
        # ax = fig.add_subplot(1,1,1) 
        # ax.set_xlabel('Principal Component 1', fontsize = 15)
        # ax.set_ylabel('Principal Component 2', fontsize = 15)
        # ax.set_title('2 component PCA', fontsize = 20)
        # ax.scatter()
        plt.show()
        # targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
        # colors = ['r', 'g', 'b']
        # for target, color in zip(targets,colors):
        #     indicesToKeep = finalDf['target'] == target
        #     ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
        #             , finalDf.loc[indicesToKeep, 'principal component 2']
        #             , c = color
        #             , s = 50)
        # ax.legend(targets)
        # ax.grid()

     
if __name__=="__main__":
    pipeline = Pipeline()
    pipeline.preProcess()
    pipeline.reduceDimension()