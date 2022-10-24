from cProfile import label
from fcmeans import FCM
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import csv

# Read in raw data
rawDataFile = './PreProcessedData/rawData.npy'
rawData = np.load(rawDataFile)

# Preprocess the raw data before PCA
inputMean = np.mean(rawData, axis = 0)
centeredData = rawData - inputMean
inputSD = np.std(centeredData, axis = 0)
zeroColumns = inputSD == 0
nonZeroSD = np.delete(inputSD, zeroColumns, axis = 0)
preprocessedData = np.delete(centeredData, zeroColumns, axis = 1)/nonZeroSD

# Apply PCA
n_comps = 10
pca = PCA(n_components = n_comps)
reducedData = pca.fit_transform(preprocessedData)

f = open('clustering_scores.csv', 'w', newline = '')
writer = csv.writer(f)

clusters_to_check = np.arange(2, 30, 1, dtype = np.int32)
SC_original_data = []
SC_reduced_data = []
# PC_original_data = []
# PEC_original_data = []
# PC_reduced_data = []
# PEC_reduced_data = []
for c in clusters_to_check:
    clm = FCM(n_clusters = c)
    clm.fit(rawData)
    labels = clm.predict(rawData)
    silhouette_coefficient_original = silhouette_score(rawData, labels)
    SC_original_data.append(silhouette_coefficient_original)
    print(f"Original data {c}: {silhouette_coefficient_original}")
    # PC_original = clm.partition_coefficient
    # PEC_original = clm.partition_entropy_coefficient
    # PC_original_data.append(PC_original)
    # PEC_original_data.append(PEC_original)
    # print(f"Original data {c}: PC = {PC_original}, PEC = {PEC_original}")

    clm = FCM(n_clusters = c)
    clm.fit(reducedData)
    labels = clm.predict(reducedData)
    silhouette_coefficient_reduced = silhouette_score(reducedData, labels)
    SC_reduced_data.append(silhouette_coefficient_reduced)
    print(f"Reduced data {c}: {silhouette_coefficient_reduced}")
    # PC_reduced = clm.partition_coefficient
    # PEC_reduced = clm.partition_entropy_coefficient
    # PC_reduced_data.append(PC_reduced)
    # PEC_reduced_data.append(PEC_reduced)
    # print(f"Reduced data {c}: PC = {PC_reduced}, PEC = {PEC_reduced}")

    writer.writerow([c, silhouette_coefficient_original, silhouette_coefficient_reduced])
    # writer.writerow([c, PC_original, PEC_original, PC_reduced, PEC_reduced])

plt.plot(clusters_to_check, SC_original_data, label = 'Original data')
plt.plot(clusters_to_check, SC_reduced_data, label = 'Reduced data')
# plt.plot(clusters_to_check, PC_original_data, label = 'PC original')
# plt.plot(clusters_to_check, PEC_original_data, label = 'PEC original')
# plt.plot(clusters_to_check, PC_reduced_data, label = 'PC reduced')
# plt.plot(clusters_to_check, PEC_reduced_data, label = 'PEC reduced')
plt.legend()
plt.xlabel("Number of clusters")
plt.ylabel("SC")
plt.xticks(clusters_to_check)
plt.show()