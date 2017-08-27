from sklearn.externals import joblib
import pandas as pd
import numpy as np


def survived_probability(req):
    model = joblib.load('rf_pipeline.gzip')
    X = pd.DataFrame([req], columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare'])
    return model.predict_proba(X)[0][1]


if __name__ == '__main__':
    req = {'Pclass': 3, 'Sex': 1, 'Age': 22.0, 'SibSp': 1, 'Parch': 0, 'Fare': 7.25}
    print 'Success: {}'.format(np.isclose(survived_probability(req), 0.07735214))
