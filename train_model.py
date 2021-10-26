import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import mlflow

# # Start mlflow run

mlflow.set_experiment('predict-rider-count')
mlflow.start_run()
mlflow.sklearn.autolog()

# # Train elasticnet linear regression model with k-folds cross validation

# Load data, split into train and test

data = pd.read_csv('bikeshare.csv')
feature_cols = ['season','yr','mnth','holiday','weekday','workingday','weathersit','temp','atemp','hum','windspeed']
label_cols = ['cnt']
data_features = data[feature_cols].values
data_labels = data[label_cols].values

train_features = data_features[:486]
train_labels = data_labels[:486]
test_features = data_features[486:]
test_labels = data_labels[486:]

# Train model

normalizer = StandardScaler()
estimator = ElasticNet(random_state = 288793)

pipeline = Pipeline([
  ('normalizer', normalizer),
  ('estimator', estimator)
])

parameters = {'estimator__l1_ratio': [.1, .5, .9, .99, 1]}

optimizer = GridSearchCV(pipeline, parameters)

model = optimizer.fit(train_features, train_labels)
model.score(train_features, train_labels)
model.score(test_features, test_labels)

# # End mlflow run

mlflow.end_run()
