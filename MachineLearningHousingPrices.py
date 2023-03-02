from sklearn.datasets import fetch_california_housing
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

dataset = fetch_california_housing()

feature_names = dataset["feature_names"]

print(f"feature_names {feature_names}")

print(dataset.data)

data_original = (dataset.data)

x_scaled = preprocessing.scale(dataset.data)

pft = preprocessing.PolynomialFeatures(degree = 2)

x_poly = pft.fit_transform(x_scaled)

