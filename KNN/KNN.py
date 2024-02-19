"""
KNN Classifier Implementation
"""

from collections import Counter
import numpy as np

def euclidean_distance(x1, x2):
    """
    Compute the Euclidean distance between two vectors
    d = root((x1-x2)^2 + (y1-y2)^2)
    """
    distance = np.sqrt(np.sum((x1-x2)**2))
    return distance

class KNN:
    """
    K Nearest Neighbors classifier
    """
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        """
        Train the model
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        """
        Predict the class for the input data
        """
        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict(self, x):
        """
        Helper function to predict single sample
        """
        # Compute the distance
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
    
        # Get the closest k
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # # Count the occurrences of each label
        # label_counts = {}
        # for label in k_nearest_labels:
        #     if label in label_counts:
        #         label_counts[label] += 1
        #     else:
        #         label_counts[label] = 1

        # # Find the label with the maximum count
        # most_common_label = max(label_counts, key=label_counts.get)

        # Majority vote
        most_common = Counter(k_nearest_labels).most_common()
        return most_common[0][0]