import pandas as pd 
import numpy as np 

class MyColumnTransformer :

    def __init__(self,transformers):
        """
        Parameters
        ----------
        transformers : list of tuples 
            Format:
            [
                ("name",transformer_object,["column1","column2"]),
                ...
            ]
        """
        self.transformers = transformers 
        self.fitted_transformers = []

    def fit(self,X):
        if not isinstance(X,pd.DataFrame):
            raise TypeError(
                "Input must be a pandas DataFrame."
            )
        self.fitted_transformers = []

        for name , transformer , columns in self.transformers:
            transformer.fit(X[columns])
            self.fitted_transformers.append(
                (name , transformer , columns)
            )

        return self 

    def transform(self,X):
        if not isinstance(X,pd.DataFrame):
            raise TypeError(
                "Input must be a pandas DataFrame."
            )
        if not self.fitted_transformers:
            raise ValueError(
                "call fit before transform"
            )
        transform_parts = []
        for name , transformer , columns in self.fitted_transformers:
            transformed = transformer.transform(X[columns])
            transform_parts.append(transformed)
        
        return pd.concat(transform_parts , axis =1)

    def fit_transform(self,X):
        self.fit(X)
        return self.transform(X)

