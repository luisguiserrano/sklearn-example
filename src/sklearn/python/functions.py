import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

def train_model(model_name = "logistic_regression"):
    dataset = pd.DataFrame({'x_1':[0,0,1,1,2,2],
                        'x_2':[0,1,0,2,1,2],
                        'y':[0,0,0,1,1,1]})

    features = np.array(dataset[['x_1', 'x_2']])
    labels = np.array(dataset['y'])

    if model_name == 'logistic_regression':
        model = LogisticRegression()
    if model_name == 'decision_tree':
        model = DecisionTreeClassifier()
    if model_name == 'svm':
        model = SVC()

    model.fit(features, labels)

    score = model.score(features, labels)

    predictions = model.predict(features)

    return predictions, score

p, s = training("logistic_regression")
print(p)
print(s)
