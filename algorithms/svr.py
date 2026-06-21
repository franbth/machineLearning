from data.loaders import load_diabetes, load_california_housing, load_energy_efficiency
from utils.metrics import show_regression_metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

DATASETS = ["california housing", "diabetes", "energy efficiency"]

def run(dataset_name):

    match dataset_name:
        case "california housing":
            x, y, feature_names, name=load_california_housing() 
            svr= SVR()
        case "diabetes":
            x, y, feature_names, name=load_diabetes()
            svr = SVR(C=100)
        case "energy efficiency":
            x, y, feature_names, name=load_energy_efficiency()
            svr= SVR()

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    print(feature_names)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)  # ojo: sin fit acá

    svr.fit(X_train_scaled, y_train)
    y_pred=svr.predict(X_test_scaled)
    show_regression_metrics(y_test, y_pred)
