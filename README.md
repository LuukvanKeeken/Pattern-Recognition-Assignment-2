# Pattern Recognition Assignment 2

Run PipelineGenes.py from the root folder to process the pipeline. A full grid search costs much time. Therefore, the grid search is skipped when the file Data/GridSearch.npy is available. The first time the code is run, it will read the data set and store it in the Data folder as rawData.npy. This is time consuming, but only done once.

Run PipelineCreditcards.py from the root folder. Above the main loop, set `number_of_rounds` to the desired number of times the experiment should be repeated. By default it is set to 100. Average accuracies and F1-scores will be printed on the command line at the end. The Final Results folder will contain all accuracies and F1-scores for each model saved to files, as well as relevant plots.

Run PipelineCreditcards.py from the root folder to process the pipeline