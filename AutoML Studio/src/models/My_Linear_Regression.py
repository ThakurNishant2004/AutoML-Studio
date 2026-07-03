import pandas as pd 
import numpy as np 
from sklearn.metrics import r2_score

class My_Linear_Regression:
    def __init__(self,learning_rate=0.01,epochs=1000):
        self.learning_rate = learning_rate 
        self.epochs = epochs 
        self.w = 0 
        self.b = 0 

    def fit(self,X,y):
        self.w = np.zeros(X.shape[1])
        self.b = 0 
        for _ in range(self.epochs):
            y_pred = np.dot(X,self.w)+self.b
            error = y_pred - y 
            dw = (2/X.shape[0])*np.dot(X.T,error)
            db = (2/X.shape[0])*np.sum(error)
            self.w -= self.learning_rate*dw 
            self.b -= self.learning_rate*db 
        return self 

    def predict(self,X):
        return  np.dot(X,self.w)+self.b  

    def score(self, X, y):
        y_pred = self.predict(X)
        return r2_score(y, y_pred)