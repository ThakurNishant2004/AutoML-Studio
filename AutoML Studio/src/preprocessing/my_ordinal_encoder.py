import numpy as np 
import pandas as pd
class MyOrdinalEncoder :
    def __init__(self,categories):
        self.categories = categories
        self.category_mapping = {}
    def fit(self,X):
        if not isinstance(X,pd.DataFrame):
            raise TypeError(
                "Input must be pandas dataframe"
            )
        
        for column , order in self.categories.items():
            self.category_mapping[column] = {
                category:idx
                for idx , category in enumerate(order)
            }

        return self 
        
    
    def transform(self,X):

        if not self.category_mapping:
            raise ValueError(
                "call fit before transform."
            )
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "Input must be a pandas DataFrame."
            )
        
        X = X.copy()
        for column in self.category_mapping:
            X[column]=X[column].map(
                self.category_mapping[column]
            )
        return X 

    
    def fit_transform(self,X):
        self.fit(X)
        return self.transform(X)