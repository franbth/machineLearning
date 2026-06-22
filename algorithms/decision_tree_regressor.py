from data.loaders import load_california_housing, load_diabetes, load_bike_sharing
from utils.metrics import show_regression_metrics
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import pandas as pd

DATASETS = ["california housing", "diabetes",  "bike sharing"]

def run(dataset_name):
    match dataset_name:
        case "california housing":
            x, y, feature_names, name=load_california_housing()  
        case "diabetes":
            x, y, feature_names, name=load_diabetes() 
        case "bike sharing":
            x, y, feature_names, name=load_bike_sharing()

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    print(feature_names)

    tree = DecisionTreeRegressor()

    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    show_regression_metrics(y_test, y_pred)

