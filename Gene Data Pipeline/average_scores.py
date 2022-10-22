import csv
import numpy as np
import matplotlib.pyplot as plt

score_sums_original = []
score_sums_reduced = []

with open('clustering_scores0.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter = ',')
    for row in csv_reader:
        score_sums_original.append(float(row[1]))
        score_sums_reduced.append(float(row[2]))


with open('clustering_scores1.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter = ',')
    line = 0
    for row in csv_reader:
        score_sums_original[line] += float(row[1])
        score_sums_reduced[line] += float(row[2])
        line += 1

with open('clustering_scores2.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter = ',')
    line = 0
    for row in csv_reader:
        score_sums_original[line] += float(row[1])
        score_sums_reduced[line] += float(row[2])
        line += 1

with open('clustering_scores3.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter = ',')
    line = 0
    for row in csv_reader:
        score_sums_original[line] += float(row[1])
        score_sums_reduced[line] += float(row[2])
        line += 1

with open('clustering_scores_5_comps.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter = ',')
    line = 0
    for row in csv_reader:
        score_sums_original[line] += float(row[1])
        score_sums_reduced[line] += float(row[2])
        line += 1

score_sums_original = np.array(score_sums_original)
print(score_sums_original/5)
score_sums_reduced = np.array(score_sums_reduced)
print(score_sums_reduced/5)

clusters_to_check = np.arange(2, 30, 1, dtype = np.int32)
plt.plot(clusters_to_check, score_sums_original, label = 'Original data')
plt.plot(clusters_to_check, score_sums_reduced, label = 'Reduced data')
plt.legend()
plt.xlabel("Number of clusters")
plt.ylabel("Silhouette Coefficient")
plt.xticks(clusters_to_check)
plt.title("Silhouette Coefficients for original and reduced data")
plt.show()