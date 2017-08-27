import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

df = pd.read_csv('data/train.csv')

X = df.loc[:, ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
# convert female=0 and male=1
X['Sex'] = X['Sex'].astype('category', categories=['female', 'male']).cat.codes

y = df['Survived'].values

model = make_pipeline(Imputer(),
                      RandomForestClassifier(max_features=1.0, max_depth=10, n_estimators=200, random_state=42))

model.fit(X, y)

joblib.dump(model, 'rf_pipeline.gzip')
