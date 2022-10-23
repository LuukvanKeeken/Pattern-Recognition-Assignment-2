import numpy as np
import pickle

GridSearchFile = './Gene Data Pipeline/Data/GridSearch.npy'
classifiers =  './Gene Data Pipeline/Data/ClassifiersGridSearch.npy'
cluster = './Gene Data Pipeline/Data/ClusterGridSearch.npy'


with open (classifiers, 'rb') as fp:
    array1 = np.array(pickle.load(fp))

with open (cluster, 'rb') as fp:
    array2 = np.array(pickle.load(fp))

newResults = []
for row in array1:
    components = row[0]
    indices2 = np.where(array2[:,0]==components)[0]
    if len(indices2) != 1:
        kjkkljln        
    row2 = array2[indices2[0]]
    addRow = np.delete(row2, 0)
    mergedRow = np.append(row, addRow)
    newResults.append(mergedRow)

with open(GridSearchFile, 'wb') as fp:
            pickle.dump(newResults, fp)