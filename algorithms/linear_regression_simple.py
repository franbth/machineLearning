from data.loaders import load_diabetes, load_california_housing, load_boston_housing
from utils.metrics import show_regression_metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

DATASETS = ["diabetes", "california housing", "boston housing"]

def run(dataset_name):

    match dataset_name:
        case "diabetes":
            x, y, feature_names, name=load_diabetes()
            x=x[:, 2].reshape(-1,1)
        case "california housing":
            x, y, feature_names, name=load_california_housing()   
            x=x[:, 0].reshape(-1,1)
        case "boston housing":
            x, y, feature_names, name=load_boston_housing()
            x=x.iloc[:, [5]]

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    print(feature_names)
    
    linReg = LinearRegression()

    linReg.fit(X_train, y_train)
    y_pred = linReg.predict(X_test)
    show_regression_metrics(y_test, y_pred)
