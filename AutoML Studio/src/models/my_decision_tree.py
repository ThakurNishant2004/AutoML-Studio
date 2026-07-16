import pandas as pd 
import numpy as np
from collections import Counter

class Node :
    def __init__(
            self ,
            feature=None ,
            threshold = None ,
            left = None ,
            right = None ,
            value = None 
            ):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value 


class MyDecisionTree:
    def __init__(
        self,
        max_depth=10,
        min_samples_split=2
    ):

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

        self.root = None

    def fit(self,X,y):
        self.root = self.build_tree(X,y)
        return self 

    def entropy(self,y):
        classes,counts = np.unique(y,return_counts=True)
        probabilities = counts/len(y)
        entropy = -np.sum(probabilities*np.log2(probabilities))
        return entropy

    def split(self,X_column,threshold):
        left_idx = np.where(
            X_column <= threshold
        )[0]

        right_idx = np.where(
            X_column > threshold

        )[0]
        
        return left_idx , right_idx

    def information_gain(
            self,
            y,
            left_idx,
            right_idx
    ):
        parent_entropy = self.entropy(y)
        n=len(y)
        n_left = len(left_idx)
        n_right = len(right_idx)
        if n_left==0 or n_right==0:
            return 0 
        
        left_entropy = self.entropy(y[left_idx])
        right_entropy = self.entropy(y[right_idx])
        child_entropy = (
            (n_left/n)*left_entropy + (n_right/n)*right_entropy
        )
        gain = parent_entropy-child_entropy
        return gain

    def build_tree(self, X, y, depth=0):

        n_samples = X.shape[0]
        n_labels = len(np.unique(y))

        # Base cases 
        if(
            n_labels ==1 
            or depth >= self.max_depth 
            or n_samples < self.min_samples_split
        ):

            leaf_value = Counter(y).most_common(1)[0][0]

            return Node(value=leaf_value)

        # Find Best Split 
        feature , threshold = self.best_split(X,y)
        
        # Split data 
        left_idx , right_idx = self.split(
            X[:, feature],
            threshold
        )

        # Recursive calls 
        left = self.build_tree(
            X[left_idx],
            y[left_idx],
            depth+1
        )

        right = self.build_tree(
            X[right_idx],
            y[right_idx],
            depth + 1
        )

        return Node(
            feature = feature,
            threshold = threshold,
            left = left,
            right = right
        )


    def best_split(self,X,y):

        best_gain = -1 
        split_index = None 
        split_threshold = None 

        n_features = X.shape[1]

        for feature in range(n_features):
            X_column = X[:,feature]
            thresholds = np.unique(X_column)

            for threshold in thresholds:
                left_idx , right_idx = self.split(
                    X_column,
                    threshold
                )
                gain = self.information_gain(
                    y,
                    left_idx,
                    right_idx
                )

                if gain>best_gain:

                    best_gain = gain
                    split_index = feature
                    split_threshold = threshold

        return split_index , split_threshold

    def traverse_tree(self, x, node):

        if node.value is not None:
            return node.value

        if x[node.feature] <= node.threshold:

            return self.traverse_tree(
                x,
                node.left
            )

        return self.traverse_tree(
            x,
            node.right
        )


    def predict(self, X):
        
        predictions = [
            self.traverse_tree(x, self.root)
            for x in X
        ]

        return np.array(predictions)

