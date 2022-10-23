from fcmeans import FCM
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import csv

# Read in raw data
#rawDataFile = './PreProcessedData/rawData.npy'
rawDataFile = './Gene Data Pipeline/Data/rawData.npy'
rawData = np.load(rawDataFile)

# Preprocess the raw data before PCA
inputMean = np.mean(rawData, axis = 0)
centeredData = rawData - inputMean
inputSD = np.std(centeredData, axis = 0)
zeroColumns = inputSD == 0
nonZeroSD = np.delete(inputSD, zeroColumns, axis = 0)
preprocessedData = np.delete(centeredData, zeroColumns, axis = 1)/nonZeroSD

# Apply PCA
n_comps = 5
pca = PCA(n_components = n_comps)
reducedData = pca.fit_transform(preprocessedData)

# Set the minimum and maximum numbers of clusters to check
minimum_number_of_clusters = 2
maximum_number_of_clusters = 30
clusters_to_check = np.arange(minimum_number_of_clusters, maximum_number_of_clusters + 1, dtype = np.int32)

# For each number of clusters to check, fit a fuzzy c-means model 
# to both the original and the reduced data. Save the silhouette
# coefficients to a file.
SC_original_data = []
SC_reduced_data = []
f = open('clustering_silhouette_scores.csv', 'w', newline = '')
writer = csv.writer(f)
for c in clusters_to_check:
    clm = FCM(n_clusters = c)
    clm.fit(rawData)
    labels = clm.predict(rawData)
    silhouette_coefficient_original = silhouette_score(rawData, labels)
    SC_original_data.append(silhouette_coefficient_original)
    print(f"Original data {c} clusters: {silhouette_coefficient_original}")

    clm = FCM(n_clusters = c)
    clm.fit(reducedData)
    labels = clm.predict(reducedData)
    silhouette_coefficient_reduced = silhouette_score(reducedData, labels)
    SC_reduced_data.append(silhouette_coefficient_reduced)
    print(f"Reduced data {c} clusters: {silhouette_coefficient_reduced}")

    writer.writerow([c, silhouette_coefficient_original, silhouette_coefficient_reduced])

# Plot the silhouette scores
plt.plot(clusters_to_check, SC_original_data, label = 'Original data')
plt.plot(clusters_to_check, SC_reduced_data, label = 'Reduced data')
plt.legend()
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Coefficient')
plt.xticks(clusters_to_check)
plt.title("Silhouette Coefficients for original and reduced data")
plt.show()