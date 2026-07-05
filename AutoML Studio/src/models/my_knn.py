import pandas as pd 
import numpy as np
from collections import Counter

class MyKNN :

    def __init__(self,k=3):
        self.k=k
        self.X_train = None 
        self.y_train = None 

    def fit(self,X,y):
        if not isinstance(X, (np.ndarray, pd.DataFrame)):
            raise TypeError(
                "X must be a NumPy array or Pandas DataFrame."
            )

        if not isinstance(y, (np.ndarray, pd.Series)):
            raise TypeError(
                "y must be a NumPy array or Pandas Series."
            )

        if isinstance(X, pd.DataFrame):
            X = X.values

        if isinstance(y, pd.Series):
            y = y.values

        self.X_train = X
        self.y_train = y

        return self

    def predict(self,X):
        if self.X_train is None:
            raise ValueError(
                "Call fit() before predict()"
            )
        if isinstance(X,pd.DataFrame):
            X = X.values

        predictions = []

        for x in X:
            distances = np.sqrt(
                np.sum(self.X_train - X)**2 
            )
            k_indices = np.argsort(distances)[:self.k]
            k_labels = self.y_train[k_indices]

            prediction = Counter(k_labels).most_common(1)[0][0]

            predictions.append(prediction)

        return np.array(predictions)
