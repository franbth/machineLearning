from sklearn import datasets
from sklearn.datasets import make_regression
import pandas as pd


#sklearn
def load_iris ():
    iris = datasets.load_iris()
    return iris.data, iris.target, iris.feature_names, "Iris"

def load_wine():
    wine = datasets.load_wine()
    return wine.data, wine.target, wine.feature_names, "Wine"

def load_breast_cancer():
    breast_cancer = datasets.load_breast_cancer()
    return breast_cancer.data, breast_cancer.target, breast_cancer.feature_names, "Breast Cancer"

def load_california_housing():
    california_housing = datasets.fetch_california_housing()
    return california_housing.data, california_housing.target, california_housing.feature_names, "California Housing"

def load_digits():
    digits = datasets.load_digits()
    return digits.data, digits.target, digits.feature_names, "Digits"

def load_diabetes():
    diabetes = datasets.load_diabetes()
    return diabetes.data, diabetes.target, diabetes.feature_names, "Diabetes"

def load_synthetic():
    n_feat =4
    X, y = make_regression(n_samples=200, n_features=n_feat, noise=10, random_state=42)
    feature_names = [f"X{i+1}" for i in range(n_feat)]
    return X, y, feature_names, "Synthetic"

#openml
def load_titanic ():
    titanic = datasets.fetch_openml(name="titanic", version=1, as_frame=True)
    return titanic.data, titanic.target, titanic.feature_names, "Titanic"

def load_boston_housing():
    boston_housing = datasets.fetch_openml(name="boston", version=1, as_frame=True)
    return boston_housing.data, boston_housing.target, boston_housing.feature_names, "Boston Housing"

def load_wine_quality():
    wine_quality = datasets.fetch_openml(name="wine-quality-red", version=1, as_frame=True)
    return wine_quality.data, wine_quality.target, wine_quality.feature_names, "Wine Quality"

def load_automobile():
    automobile = datasets.fetch_openml(name="autos", version=1, as_frame=True)
    automobile.target=automobile.data["price"]
    automobile.data=automobile.data.drop(columns="price")
    return automobile.data, automobile.target, automobile.feature_names, "Automobile"

def load_energy_efficiency():
    energy_efficiency = datasets.fetch_openml(name="energy-efficiency", version=1, as_frame=True, target_column="y1")
    energy_efficiency.data=energy_efficiency.data.drop(columns="y2")
    energy_efficiency.target=pd.to_numeric(energy_efficiency.target)
    return energy_efficiency.data, energy_efficiency.target, energy_efficiency.feature_names, "Energy Efficiency"

def load_bike_sharing():
    bike_sharing = datasets.fetch_openml(name="Bike_Sharing_Demand", version=2, as_frame=True)
    print(bike_sharing.frame.columns.tolist())
    print(bike_sharing.frame.dtypes)
    return bike_sharing.data, bike_sharing.target, bike_sharing.feature_names, "Bike Sharing"

def load_air_quality():
    air_quality = datasets.fetch_openml(name="air-quality", version=1, as_frame=True)
    return air_quality.data, air_quality.target, air_quality.feature_names, "Air Quality"

def load_heart_disease():
    heart_disease = datasets.fetch_openml(name="heart-disease", version=1, as_frame=True)
    return heart_disease.data, heart_disease.target, heart_disease.feature_names, "Heart Disease"

def load_20newsgroups():
    newsgroups = datasets.fetch_20newsgroups(subset='all')
    return newsgroups.data, newsgroups.target, None, "20 Newsgroups"
