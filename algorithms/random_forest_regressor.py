from data.loaders import load_california_housing, load_diabetes, load_air_quality
from utils.metrics import show_regression_metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

DATASETS = ["california housing", "diabetes",  "air quality"]

def run(dataset_name):
    match dataset_name:
        case "california housing":
            x, y, feature_names, name=load_california_housing() 
            forest = RandomForestRegressor() #se puede ajustar n_estimators=nro de arboles
        case "diabetes":
            x, y, feature_names, name=load_diabetes() 
            forest = RandomForestRegressor() #se puede ajustar max_depth=nro de nodos por arbol
        case "air quality":
            x, y, feature_names, name=load_air_quality()
            forest = RandomForestRegressor()

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    print(feature_names)

    forest.fit(X_train, y_train)
    y_pred=forest.predict(X_test)
    show_regression_metrics(y_test, y_pred)