import pandas as pd 
import numpy as np 
from sklearn.metrics import accuracy_score

class MyLogisticRegression :

    def __init__(self,learning_rate=0.01,epochs=1000):
        self.learning_rate = learning_rate 
        self.epochs = epochs 
        self.w = None
        self.b = None
    
    def sigmoid(self,z):
        return 1/(1+np.exp(-z))

    def fit(self , X, y):
        self.w = np.zeros(X.shape[1])
        self.b = 0
        for _ in range(self.epochs) :
            z = np.dot(X,self.w)+self.b 
            y_pred = self.sigmoid(z)
            error = y_pred - y 
            n = X.shape[0]
            dw = (1/n)*np.dot(X.T,error)
            db = (1/n)*np.sum(error)
            self.w = self.w-self.learning_rate*dw
            self.b = self.b-self.learning_rate*db
        return self

    def predict_proba(self, X):
        if self.w is None:
            raise ValueError(
                "Call fit() before predict()."
            )
        z = np.dot(X, self.w) + self.b
        return self.sigmoid(z)

    def predict(self,X):
        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(int)

    def score(self,X,y):
        return accuracy_score(
            y,
            self.predict(X)
        )