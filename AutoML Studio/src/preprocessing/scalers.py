import numpy as np 
import pandas as pd
# Input validation -> isinstance 
class MyStandardScaler :
    def __init__(self):
        self.mean=None
        self.std = None
    def fit(self , X ):
        # Checking whether X is a NumPy array or a DataFrame
        is_numpy = isinstance(X,np.ndarray)
        is_dataframe = isinstance(X,pd.DataFrame)
        if not(is_numpy or is_dataframe):
            raise TypeError(
                "Input must be a Numpy array or Pandas DataFrame"
            )
        if is_numpy:
            self.mean = X.mean(axis=0)
            self.std = X.std(axis=0)
        else :
            self.mean = X.mean()
            self.std = X.std()
        return self 
    def transform(self,X):
        if self.mean is None or self.std is None:
            raise ValueError(
                "Call fit() before transform()."
            ) 
        return (X-self.mean)/self.std 
    def fit_transform(self,X):
        self.fit(X)
        return self.transform(X)