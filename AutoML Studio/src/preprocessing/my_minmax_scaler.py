import pandas  as pd 
import numpy as np 
class My_MinMax_Scaler:
    def __init__(self):
        self.min = None 
        self.max = None
    def fit(self,X):
        if not (isinstance(X,pd.DataFrame) or isinstance(X,np.ndarray)):
            raise TypeError(
                "Input must be ndarrya or DataFrame"
            )
        self.min = X.min()
        self.max = X.max()
        return self
    def transform(self,X):
        if not (isinstance(X,pd.DataFrame) or isinstance(X,np.ndarray)):
            raise TypeError(
                "Input must be ndarrya or DataFrame"
            )
        if self.min is None or self.max is None:
            raise ValueError(
                "Call fit before transform"
            )
        X = X.copy()
        for column in X.columns:
            X[column]=(X[column]-self.min[column])/(self.max[column]-self.min[column])
        return X 

    def fit_transform(self,X):
        self.fit(X)
        return self.transform(X)