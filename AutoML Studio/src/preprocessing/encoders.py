import pandas as pd 
import numpy as np 

class MyOneHotEncoder :
    def __init__(self,handle_unknown = "error"):
        self.handle_unknown = handle_unknown
        self.categories = {}
    def fit (self,X):
        isNumpy = isinstance(X,np.ndarray)
        isDataFrame = isinstance(X,pd.DataFrame)
        if not(isNumpy or isDataFrame):
            raise TypeError(
                "Input must be a numpy array or pandas dataframe."
            )
        if isNumpy :
            X = pd.DataFrame(X)

        for column in X.columns:
            self.categories[column] = np.unique(X[column])

        return self 
    def transform(self,X):
        if not self.categories:
            raise ValueError(
                "Call fit() before transform()."
            )
        isNumpy = isinstance(X,np.ndarray)
        isDataFrame = isinstance(X,pd.DataFrame)
        if not(isNumpy or isDataFrame):
            raise TypeError(
                "Input must be a numpy array or pandas dataframe."
            )
        if isNumpy :
            X = pd.DataFrame(X)
        
        encoded_df = pd.DataFrame(index=X.index)

        for column in X.columns:
            if column  not in self.categories:
                raise ValueError(f"Column '{column}' was not seen during fit().")
            
            for category in self.categories[column]:
                new_column = f"{column}_{category}"
                encoded_df[new_column]=(X[column]==category).astype(int)

        return encoded_df 
    def fit_transform(self,X):
        self.fit(X)
        return self.transform(X)