import os
import pickle
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def dataSetSummary():
    # count number of samples per class
    samplesPerClass = [0]*len(labelNames)
    for label in labels:
        samplesPerClass[label] +=1
    leastItems = np.min(samplesPerClass)

    classWeights={}
    for index, i in enumerate(samplesPerClass):
        weight = leastItems/i
        classWeights[index]=weight
    
    print("The data set has "+ str(len(labels))+ " samples divided over " + str(len(labelNames)) + " classes.")
    print("For labels " + str(labelNames) + ",the number of samples per class are: " + str(samplesPerClass))
    print("Number of features per sample: "+ str(rawData.shape[1]))
    
    zeroColumns = 0
    for column in rawData.T:
        if len(column) == np.count_nonzero(column==0.0):
            zeroColumns +=1
    print("However, of those features, "+ str(zeroColumns)+ " features are zero for each sample")
    print()

def PcaAnalysis():
    n_comps = len(normalizedX)
    pca = PCA(n_components=n_comps)
    pca.fit(normalizedX)
    exp_var = pca.explained_variance_ratio_ * 100
    cum_exp_var = np.cumsum(exp_var)

    # More versatile wrapper
    fig, host = plt.subplots(figsize=(8,5)) # (width, height) in inches
        
    par1 = host.twinx()
        
    host.set_xlim(0, n_comps)
    host.set_ylim(0, 11)
    par1.set_ylim(0, 110)
        
    host.set_xlabel("Principal components")
    host.set_ylabel("Individual explained variance")
    par1.set_ylabel("Cumulative explained variance")
    p1 = host.bar(range(1, n_comps+1), exp_var, align='center', label='Individual explained variance')
    p2, = par1.step(range(1, n_comps+1), cum_exp_var, where='mid', label='Cumulative explained variance', color='red')

    lns = [p1, p2]
    host.legend(handles=lns, loc='upper left')

    fig.tight_layout()
    plt.savefig("./Figures/GenesPcaVariance.png")

def featureExtraction(numberOfComponents):
    # TODO: als we dit in het report gebruiken dan iedere data point de kleur van de class geven.
    pca = PCA(numberOfComponents)
    reducedDimensionsData = pca.fit_transform(normalizedX)

    if (numberOfComponents <= 3):
        fig = plt.figure(figsize = (8,8))
        if numberOfComponents == 3:
            ax = fig.add_subplot(projection='3d')
        else:
            ax = fig.add_subplot()
            
        for index, label in enumerate(labelNames):
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
        plt.savefig(f"Figures{os.sep}GenesVisualization")

def plotGridAccuracy(performances):
    fig = plt.figure(figsize = (8,8))
    xAxis = performances[:,0].astype('float64')
    performanceKnn = performances[:,1].astype('float64')
    performanceLr = performances[:,3].astype('float64')

    fig, ax = plt.subplots(figsize=(5,4))
    ax.plot(xAxis, performanceKnn, label="KNN")
    ax.plot(xAxis, performanceLr, label="LR")

    # inset axes....
    axins = ax.inset_axes([0.25, 0.3, 0.72, 0.47])
    axins.plot(xAxis, performanceKnn, label="KNN")
    axins.plot(xAxis, performanceLr, label="LR")

    # sub region of the original image
    x1, x2, y1, y2 = 2, 13, 0.85, 1.005
    axins.set_xlim(x1, x2)
    axins.xaxis.set_major_locator(MaxNLocator(integer=True))
    axins.set_ylim(y1, y2)
    #axins.set_xticklabels([])
    #axins.set_yticklabels([])

    ax.indicate_inset_zoom(axins, edgecolor="black")

    # plt.plot(xAxis, performanceKnn, label="KNN")
    # plt.plot(xAxis, performanceLr, label="LR")
    # ax = fig.gca()
    # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # plt.xticks(xAxis)
    plt.xlabel("PCA dimensions")
    plt.ylabel("Accuracy")
    # plt.ylim((0.85,1.05))
    plt.xlim((1,170))
    plt.legend()
    plt.title("Best model accuracies with PCA components")
    figureName =f"Figures{os.sep}GenesClassifiersGridSearch" 
    plt.savefig(figureName)


if __name__=="__main__":
    # File locations
    rawDataFile = './Gene Data Pipeline/Data/rawData.npy'
    labelsFile = './Gene Data Pipeline/Data/labels.npy'
    labelsNameFile = './Gene Data Pipeline/Data/labelNames.npy'
    GridSearchClassifiersFile = './Gene Data Pipeline/Data/ClassifiersGridSearch.npy'

    # Load data
    rawData = np.load(rawDataFile)
    labels = np.load(labelsFile)
    labelNames = np.load(labelsNameFile, allow_pickle=True)

    # Normalize the whole data set
    dataCentered = rawData - np.mean(rawData, axis=0) 
    dataSD = np.std(dataCentered, axis=0)
    zeroColumns = dataSD == 0
    normalizedX = np.delete(dataCentered,zeroColumns,axis=1)/np.delete(dataSD,zeroColumns,axis=0)

    # Plot figures conserning the data set exploration
    dataSetSummary()
    PcaAnalysis()
    featureExtraction(2)

    with open (GridSearchClassifiersFile, 'rb') as fp:
        results = np.array(pickle.load(fp))   
    results=results[results[:,0].argsort()]
    plotGridAccuracy(results)