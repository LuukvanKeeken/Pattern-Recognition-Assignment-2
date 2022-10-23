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
    xAxis = performances[:,0].astype('float64')
    performanceKnn = performances[:,1].astype('float64')
    performanceLr = performances[:,3].astype('float64')
    performanceBayes = performances[:,5].astype('float64')
    performanceFCMeans = performances[:,7].astype('float64')

    fig, ax = plt.subplots(figsize=(12,7))
    ax.plot(xAxis, performanceKnn, label="KNN")
    ax.plot(xAxis, performanceLr, label="LR")
    ax.plot(xAxis, performanceBayes, label="Bayes")
    ax.plot(xAxis, performanceFCMeans, label="FC-means")

    zoomPcaMin = 1
    zoomPcaMax = 15
    knnZoomPerformance = performanceKnn[zoomPcaMin:zoomPcaMax]
    lrZoomPerformance = performanceLr[zoomPcaMin:zoomPcaMax]
    bayesZoomPerformance = performanceBayes[zoomPcaMin:zoomPcaMax]
    fcMeansZoomPerformance = performanceFCMeans[zoomPcaMin:zoomPcaMax]
    knnIndexMax = np.argmax(knnZoomPerformance)+zoomPcaMin
    lrIndexMax = np.argmax(lrZoomPerformance)+zoomPcaMin
    bayesIndexMax = np.argmax(bayesZoomPerformance)+zoomPcaMin
    fcMeansIndexMax = np.argmax(fcMeansZoomPerformance)+zoomPcaMin

    axins = ax.inset_axes([0.1, 0.1, 0.8, 0.47])
    axins.plot(xAxis-1, performanceKnn, label="KNN", color='blue')
    axins.plot((knnIndexMax,knnIndexMax), (performanceKnn[knnIndexMax],0),color='blue',linestyle=':')
    axins.plot(xAxis-1, performanceLr, label="LR", color = 'orange')
    axins.plot((lrIndexMax,lrIndexMax), (performanceLr[lrIndexMax],0),color='orange',linestyle=':')
    axins.plot(xAxis-1, performanceBayes, label="Bayes", color='green')
    axins.plot((bayesIndexMax,bayesIndexMax), (performanceBayes[bayesIndexMax],0),color='green',linestyle=':')
    axins.plot(xAxis-1, performanceFCMeans, label="FC-means")#, color='green')
    axins.plot((fcMeansIndexMax,fcMeansIndexMax), (performanceFCMeans[fcMeansIndexMax],0),color='green',linestyle=':')
  
    #axins.axvline(knnIndexMax)
    
    #testX = knnIndexMax
    #testY = performanceKnn[knnIndexMax]
    #axins.plot(testX,testY, marker="o", markersize=5, markerfacecolor = "black")


    # sub region of the original image
    x1, x2, y1, y2 = zoomPcaMin, zoomPcaMax, 0.85, 1.005
    axins.set_xlim(x1, x2)
    axins.xaxis.set_major_locator(MaxNLocator(integer=True))
    axins.xaxis.set_ticks(range(zoomPcaMin,zoomPcaMax+1,1))
    axins.set_ylim(y1, y2)
    ax.indicate_inset_zoom(axins, edgecolor="black")

    plt.xlabel("PCA dimensions")
    plt.ylabel("Accuracy")
    plt.xlim((1,170))
    #plt.legend()
    plt.legend(bbox_to_anchor=(0.85,0.78))
    plt.title("Best model accuracies with PCA components")
    figureName =f"Figures{os.sep}GenesClassifiersGridSearch" 
    plt.savefig(figureName, dpi = 300, bbox_inches='tight')


if __name__=="__main__":
    # File locations
    rawDataFile = './Gene Data Pipeline/Data/rawData.npy'
    labelsFile = './Gene Data Pipeline/Data/labels.npy'
    labelsNameFile = './Gene Data Pipeline/Data/labelNames.npy'
    #GridSearchClassifiersFile = './Gene Data Pipeline/Data/ClassifiersGridSearch.npy'
    GridSearchFile = './Gene Data Pipeline/Data/GridSearch.npy'

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

    with open (GridSearchFile, 'rb') as fp:
        results = np.array(pickle.load(fp))   
    results=results[results[:,0].argsort()]
    if results[len(results)-1][0]>10000:
        results = np.delete(results, len(results)-1, axis=0)
    plotGridAccuracy(results)