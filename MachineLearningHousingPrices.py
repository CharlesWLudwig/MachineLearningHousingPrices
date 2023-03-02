from sklearn.datasets import fetch_california_housing
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

dataset = fetch_california_housing()

feature_names = dataset["feature_names"]

print(f"feature_names {feature_names}")

# print(dataset.data)

data_original = (dataset.data)

x_scaled = preprocessing.scale(dataset.data)

pft = preprocessing.PolynomialFeatures(degree = 2)

x_poly = pft.fit_transform(x_scaled)

X_train, x_test, y_train, y_test = train_test_split(x_poly, dataset.target, test_size= 0.40, random_state = 42)

model = linear_model.Ridge(alpha = 300)

model.train(X_train, y_train)

predictionTestSet = model.predict(x_test)

errorTestSet = mean_squared_error(y_test, predictionTestSet)

