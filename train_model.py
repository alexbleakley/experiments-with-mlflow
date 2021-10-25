import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
import mlflow

# # Start mlflow run

mlflow.set_experiment('bikeshare_v1')
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

parameters = {'l1_ratio': [.1, .5, .7, .9, .95, .99, 1]}
unfitted_estimator = ElasticNet(normalize = True, random_state = 288793)
optimizer = GridSearchCV(unfitted_estimator, parameters)
fitted_estimator = optimizer.fit(train_features, train_labels)

# # End mlflow run

mlflow.end_run()