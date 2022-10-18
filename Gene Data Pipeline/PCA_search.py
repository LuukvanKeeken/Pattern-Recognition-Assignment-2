import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

preProcessedDataFile = '../PreProcessedData/preProcessedData.npy'
preProcessedLabelsFile = '../PreProcessedData/preProcessedLabels.npy'

preProcessedData = np.load(preProcessedDataFile)
preProcessedLabels = np.load(preProcessedLabelsFile)
print(preProcessedData)

n_comps = 20
pca = PCA(n_components=n_comps)
pca.fit(preProcessedData)

exp_var = pca.explained_variance_ratio_ * 100
print(exp_var)
print(len(exp_var))
cum_exp_var = np.cumsum(exp_var)

plt.bar(range(1, n_comps+1), exp_var, align='center',
        label='Individual explained variance')

plt.step(range(1, n_comps+1), cum_exp_var, where='mid',
         label='Cumulative explained variance', color='red')

plt.ylabel('Explained variance percentage')
plt.xlabel('Principal component index')
plt.xticks(ticks=np.arange(1,n_comps+1))
plt.legend(loc='best')
plt.tight_layout()

plt.savefig("Barplot.png")