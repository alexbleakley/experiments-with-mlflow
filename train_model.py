import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn


# # Start mlflow run

mlflow.set_experiment('bikeshare')
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

# one_hot_encoder = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')
# polynomial_features = PolynomialFeatures(degree=2)
normalizer = StandardScaler()
estimator = ElasticNet(random_state = 288793)

pipeline = Pipeline([
#  ('one_hot_encoder', one_hot_encoder),
#  ('polynomial_features', polynomial_features),
  ('normalizer', normalizer),
  ('estimator', estimator)
])

parameters = {'estimator__l1_ratio': [.1, .5, .9, .99, 1]}

optimizer = GridSearchCV(pipeline, parameters)

model = optimizer.fit(train_features, train_labels)
train_score = model.score(train_features, train_labels)
mlflow.log_metric('train_score', train_score)
test_score = model.score(test_features, test_labels)
mlflow.log_metric('test_score', test_score)

# # End mlflow run

mlflow.end_run()
