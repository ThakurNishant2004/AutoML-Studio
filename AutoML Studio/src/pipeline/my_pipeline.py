import pandas as pd 
import numpy as np 

class MyPipeline :
    """
        Custom ML pipeline for sequential preprocessing and model training.

        Steps:
        Fit → Transform → Train
        Predict → Transform → Predict
    """
    def __init__(self,steps):
        self.steps = steps 
        self.fitted_steps = []

    def fit(self,X,y):
        current_X = X 
        for name , step in self.steps[:-1]:
            current_X = step.fit_transform(current_X)
            self.fitted_steps.append(
                (name,step)
            )
        
        model_name , model = self.steps[-1]
        model.fit(current_X,y)
        self.fitted_steps.append(
            (model_name,model)
        )

        return self 

    def predict(self,X):
        current_X = X 
        for name , step in self.fitted_steps[:-1]:
            current_X = step.transform(current_X)
        
        model_name , model = self.fitted_steps[-1]

        return model.predict(current_X)
