import numpy as np 
import pandas as pd
from scipy.stats import mode
class MySimpleImputer :
    def __init__(self,strategy='mean') :
        self.strategy = strategy
        self.statistic = None
    def fit(self,X):
        isNumpy = isinstance(X,np.ndarray)
        isDataframe = isinstance(X,pd.DataFrame)
        if not(isNumpy or isDataframe):
            raise TypeError(
                "Input must be a numpy array or pandas dataframe."
            )
        if isNumpy:
            if self.strategy == "mean":
                self.statistic = np.nanmean(X,axis=0)
            elif self.strategy == "median":
                self.statistic = np.nanmedian(X,axis=0)
            elif self.strategy == "most_frequent":
                self.statistic = mode(X,axis=0,nan_policy="omit",keepdims=False).mode
            else :
                raise ValueError(
                    "strategy must be 'mean', 'median', or 'most_frequent'."
                )
        else :
            if self.strategy == "mean":
                self.statistic = X.mean(skipna = True)
            elif self.strategy == "median":
                self.statistic = X.median(skipna = True)
            elif self.strategy == "most_frequent":
                self.statistic = X.mode(dropna=False).iloc[0]
            else :
                raise ValueError(
                    "strategy must be 'mean', 'median', or 'most_frequent'."
                )
        return self 
    def transform(self,X):
        if self.statistic is None:
            raise ValueError(
                "call fit() before transform()."
            )
        
        if isinstance(X,pd.DataFrame):
            return X.fillna(self.statistic)
        
        elif isinstance(X,np.ndarray):
            X = X.copy()
            mask = np.isnan(X)
            X[mask]= np.take(self.statistic,np.where(mask)[1])
            return X 
        
        else:
            raise TypeError(
                "Input must be a Numpy array or pandas DataFrame."
            )

    def fit_transform(self,X):
        self.fit(X)
        return self.transform(X)
