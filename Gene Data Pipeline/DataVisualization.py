import os
import pickle
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

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
    pca = PCA(1)
    oneDimensionData = pca.fit_transform(normalizedX)
    pca = PCA(2)
    twoDimensionData = pca.fit_transform(normalizedX)
    pca = PCA(3)
    threeDimensionData = pca.fit_transform(normalizedX)

    fig = plt.figure(figsize = (12,3))
    ax1 = fig.add_subplot(1,3,1)
    ax2 = fig.add_subplot(1,3,2)
    ax3 = fig.add_subplot(1,3,3, projection='3d')

    #     if numberOfComponents == 3:
    #         ax = fig.add_subplot(projection='3d')
    #     else:
    #         


    #fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.suptitle('Projection of the data set to different PCA components',y=1.05)
    # ax1.plot(x, y)
    # ax2.plot(x, -y)

    for index, label in enumerate(labelNames):
        indicesOfClass = labels == index
        pointsOne = oneDimensionData[indicesOfClass]
        pointsTwo = twoDimensionData[indicesOfClass]
        pointsThree = threeDimensionData[indicesOfClass]
        ax1.hist(pointsOne, alpha=0.5)
        ax2.scatter(pointsTwo[:,0],pointsTwo[:,1])
        ax3.scatter(pointsThree[:,0],pointsThree[:,1],pointsThree[:,2],label=str(index))
    ax1.set(title = "1 component", xlabel='Feature 1')
    ax2.set(title = "2 components", xlabel='Feature 1', ylabel='Feature 2')
    ax3.set(title = "3 components", xlabel='Feature 1', ylabel='Feature 2', zlabel='Feature 3')
    
    plt.subplots_adjust( wspace=0.3)
    #plt.legend()

    # pca = PCA(numberOfComponents)
    # reducedDimensionsData = pca.fit_transform(normalizedX)

    # if (numberOfComponents <= 3):
    #     fig = plt.figure(figsize = (8,8))
    #     if numberOfComponents == 3:
    #         ax = fig.add_subplot(projection='3d')
    #     else:
    #         ax = fig.add_subplot()
            
    #     for index, label in enumerate(labelNames):
    #         indicesOfClass = labels == index
    #         points = reducedDimensionsData[indicesOfClass] 
            
    #         if numberOfComponents == 1:
    #             ax.hist(points, alpha=0.5)
    #         elif numberOfComponents == 2:
    #             ax.scatter(points[0],points[1])
    #         else:
    #             ax.scatter(points[0],points[1],points[2])
    #     plt.xlabel("Principal component 1")
    #     plt.ylabel("Principal component 2")
    #plt.title("PCA")
    figureName = f"Figures{os.sep}GenesVisualization"
    plt.savefig(figureName, dpi = 300, bbox_inches='tight')

def plotClassifiersGridPerformance(performances):
    xAxis = performances[:,0].astype('float64')
    performanceKnn = performances[:,1].astype('float64')
    performanceLr = performances[:,3].astype('float64')
    performanceBayes = performances[:,5].astype('float64')

    fig, ax = plt.subplots(figsize=(12,7))
    ax.plot(xAxis, performanceKnn, label="KNN")
    ax.plot(xAxis, performanceLr, label="LR")
    ax.plot(xAxis, performanceBayes, label="Bayes")

    zoomPcaMin = 1
    zoomPcaMax = 15
    knnZoomPerformance = performanceKnn[zoomPcaMin:zoomPcaMax]
    lrZoomPerformance = performanceLr[zoomPcaMin:zoomPcaMax]
    bayesZoomPerformance = performanceBayes[zoomPcaMin:zoomPcaMax]

    knnIndexMax = np.argmax(knnZoomPerformance)+zoomPcaMin+1
    lrIndexMax = np.argmax(lrZoomPerformance)+zoomPcaMin+1
    bayesIndexMax = np.argmax(bayesZoomPerformance)+zoomPcaMin+1

    axins = ax.inset_axes([0.1, 0.1, 0.8, 0.47])
    axins.plot(xAxis, performanceKnn, label="KNN", color='blue')
    axins.plot((knnIndexMax,knnIndexMax), (performanceKnn[knnIndexMax],0),color='blue',linestyle=':')
    axins.plot(xAxis, performanceLr, label="LR", color = 'orange')
    axins.plot((lrIndexMax,lrIndexMax), (performanceLr[lrIndexMax],0),color='orange',linestyle=':')
    axins.plot(xAxis, performanceBayes, label="Bayes", color='green')
    axins.plot((bayesIndexMax,bayesIndexMax), (performanceBayes[bayesIndexMax],0),color='green',linestyle=':')

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
    plt.legend(bbox_to_anchor=(0.85,0.78))
    plt.title("Classifiers performance for different PCA components")
    figureName =f"Figures{os.sep}GenesClassifiersGridSearch" 
    plt.savefig(figureName, dpi = 300, bbox_inches='tight')

def plotClusterGridPerformance(performances):
    xAxis = performances[:,0].astype('float64')
    performanceFCMeans = performances[:,7].astype('float64')

    fig, ax = plt.subplots(figsize=(12,7))
    ax.plot(xAxis, performanceFCMeans, label="FC-means")

    zoomPcaMin = 1
    zoomPcaMax = 18

    axins = ax.inset_axes([0.15, 0.45, 0.8, 0.47])
    axins.plot(xAxis, performanceFCMeans, label="FC-means")

    # sub region of the original image
    x1, x2, y1, y2 = zoomPcaMin, zoomPcaMax, 0.1, 0.85
    axins.set_xlim(x1, x2)
    axins.xaxis.set_major_locator(MaxNLocator(integer=True))
    axins.xaxis.set_ticks(range(zoomPcaMin,zoomPcaMax+1,1))
    axins.set_ylim(y1, y2)
    ax.indicate_inset_zoom(axins, edgecolor="black")

    plt.xlabel("PCA dimensions")
    plt.ylabel("Silhouette Coefficient")
    plt.xlim((1,170))
    plt.title("Cluster performance for different PCA components")
    figureName =f"Figures{os.sep}GenesClusterGridSearch" 
    plt.savefig(figureName, dpi = 300, bbox_inches='tight')

def plotConfusionMatrix(name, dimensions, augmented, predictions):
    title = "Confusion matrix model " + name
    fileName = "GenesConfusionMatrix"+name

    if augmented:
        title += " on the augmented"
        fileName += "Augmented"
    else:
        title += " on the default"
        fileName += "Default"

    if dimensions < 200:
        title += " best-reduced data set"
        fileName += "Pca"
    else:
        title += " original data set"
        fileName += "Original"

    confusionMatrix = confusion_matrix(predictions[:,0],predictions[:,1])
    #print(title + " Precision: "+str(round(precision_score(predictions[:,0],predictions[:,1], average='weighted'),3)))
    #print(title + " Recall: "+str(round(recall_score(predictions[:,0],predictions[:,1], average='weighted'),3)))
    print(title + " F1 Score: "+str(round(f1_score(predictions[:,0],predictions[:,1], average='weighted'),3)))

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(confusionMatrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(confusionMatrix.shape[0]):
        for j in range(confusionMatrix.shape[1]):
            ax.text(x=j, y=i,s=confusionMatrix[i, j], va='center', ha='center', size='xx-large')
    plt.xticks([0,1,2,3,4],labelNames)
    plt.yticks([0,1,2,3,4],labelNames)
    
    plt.xlabel('Predictions')
    plt.ylabel('Actuals')
    plt.title(title,y=1.05)
    figureName =f"Figures{os.sep}"+fileName
    plt.savefig(figureName, dpi = 300, bbox_inches='tight')

def plotEvaluationResults(evaluationResults):
    for result in evaluationResults:
        [name, dimensions, _, _, _, standardPredictions, _, augmentedPredictions]=result
        # predictions is [yTrue, yPredicted]
        if name != "FC-means":
            plotConfusionMatrix(name, dimensions, False, standardPredictions)
            plotConfusionMatrix(name, dimensions, True, augmentedPredictions)


if __name__=="__main__":
    # File locations
    rawDataFile = './Gene Data Pipeline/Data/rawData.npy'
    labelsFile = './Gene Data Pipeline/Data/labels.npy'
    labelsNameFile = './Gene Data Pipeline/Data/labelNames.npy'
    GridSearchFile = './Gene Data Pipeline/Data/GridSearch.npy'
    evaluationResultsFile = './Gene Data Pipeline/Data/EvaluationResults.npy'

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
    plotClassifiersGridPerformance(results)
    plotClusterGridPerformance(results)

    with open (evaluationResultsFile, 'rb') as fp:
        evaluationResults = np.array(pickle.load(fp))  
    
    plotEvaluationResults(evaluationResults)