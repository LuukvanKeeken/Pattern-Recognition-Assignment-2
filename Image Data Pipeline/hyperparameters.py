hyperparameters = {
    "unreduced": {
        "clusters_2": 5,
        "rf_metric": "gini",
        "estimators": 170,
        "distance": "cosine",
        "kernel": "rbf",
        "C": 1.8,
        "K": 10,
        "weights": "uniform"
    },
    "reduced": {
        "keypoints": {
            "svm": 35,
            "rf": 35,
            "knn": 40,
            "km": 35
        },
        "clusters_1": {
            "svm": 25,
            "rf": 35,
            "knn": 45,
            "km": 25,
        },
        "clusters_2": 5,
        "rf_metric": "gini",
        "estimators": 140,
        "distance": "cosine",
        "kernel": "poly",
        "C": 0.7,
        "K": 16,
        "weights": "uniform"
    },
    "augmented": {
        "keypoints": {
            "svm": 25,
            "rf": 35,
            "knn": 25,
            "km": 25,
        },
        "clusters_1": {
            "svm": 40,
            "rf": 30,
            "knn": 25,
            "km": 40,
        },
        "clusters_2": 5,
        "rf_metric": "entropy",
        "estimators": 200,
        "distance": "manhattan",
        "kernel": "rbf",
        "C": 2,
        "K": 10,
        "weights": "uniform"
    },
}