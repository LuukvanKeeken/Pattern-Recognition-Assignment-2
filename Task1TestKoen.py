import numpy as np
from numpy import genfromtxt
from os import path

dataFileName = './Data/Genes/data.csv'
labelsFileName = './Data/Genes/labels.csv'
rawDataFile = './PreProcessedData/rawData.npy'
rawLabelsFile = './PreProcessedData/rawLabels.npy'


if (path.exists(rawDataFile)==False and path.exists(rawLabelsFile)==False):
    rawData = genfromtxt(dataFileName, delimiter=',')
    rawLabels = genfromtxt(labelsFileName, delimiter=',')
    np.save(rawDataFile, rawData)
    np.save(rawLabelsFile, rawLabels)
else:
    rawData = np.fromfile(rawDataFile)
    rawLabels = np.fromfile(rawLabelsFile)

print()