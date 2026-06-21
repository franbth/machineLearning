from data.loaders import load_california_housing, load_diabetes, load_synthetic
from utils.metrics import show_regression_metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

import pandas as pd

DATASETS = ["california housing", "diabetes",  "synthetic"]

def run(dataset_name):

    match dataset_name:
        case "california housing":
            x, y, feature_names, name=load_california_housing()  
        case "diabetes":
            x, y, feature_names, name=load_diabetes() 
        case "synthetic":
            x, y, feature_names, name=load_synthetic()

    poly = PolynomialFeatures(degree=2)
    x = poly.fit_transform(x)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    print(feature_names)
    
    linReg = LinearRegression()

    linReg.fit(X_train, y_train)
    y_pred = linReg.predict(X_test)
    show_regression_metrics(y_test, y_pred)