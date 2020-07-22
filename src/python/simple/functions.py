import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import json
from simple.utils import NumpyArrayEncoder

def train_model():
    dataset = pd.DataFrame({'x_1':[0,0,1,1,2,2],
                        'x_2':[0,1,0,2,1,2],
                        'y':[0,0,0,1,1,1]})

    features = np.array(dataset[['x_1', 'x_2']])
    labels = np.array(dataset['y'])

    model = LogisticRegression()

    model.fit(features, labels)

    predictions = np.array(model.predict(features))

    return json.dumps(predictions, cls=NumpyArrayEncoder)
