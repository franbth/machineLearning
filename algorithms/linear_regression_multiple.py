from data.loaders import load_california_housing, load_wine_quality, load_automobile
from utils.metrics import show_regression_metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd

DATASETS = ["california housing", "wine quality",  "automobile"]

def run(dataset_name):

    match dataset_name:
        case "california housing":
            x, y, feature_names, name=load_california_housing()  
        case "wine quality":
            x, y, feature_names, name=load_wine_quality() 
        case "automobile":
            x, y, feature_names, name=load_automobile()
            x = x.select_dtypes(include='number')
            combinado = pd.concat([x,y], axis=1)
            combinado = combinado.dropna()
            x=combinado.drop(columns="price")
            y=combinado["price"]



    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    print(feature_names)
    
    linReg = LinearRegression()

    linReg.fit(X_train, y_train)
    y_pred = linReg.predict(X_test)
    show_regression_metrics(y_test, y_pred)
