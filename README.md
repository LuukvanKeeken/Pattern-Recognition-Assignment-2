# Pattern Recognition Assignment 2


Run PipelineGenes.py from the root folder to process the pipeline. A full grid search costs much time. Therefore, the grid search is skipped when the file Data/GridSearch.npy is available. The first time the code is run, it will read the data set and store it in the Data folder as rawData.npy. This is time consuming, but only done once.