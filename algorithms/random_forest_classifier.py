from data.loaders import load_breast_cancer, load_wine, load_digits
from utils.metrics import show_classification_metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

DATASETS = ["breast cancer", "wine",  "digits"]

def run(dataset_name):
    match dataset_name:
        case "breast cancer":
            x, y, feature_names, name=load_breast_cancer() 
            forest = RandomForestClassifier() #se puede ajustar n_estimators=nro de arboles
        case "wine":
            x, y, feature_names, name=load_wine() 
            forest = RandomForestClassifier() #se puede ajustar max_depth=nro de nodos por arbol
        case "digits":
            x, y, feature_names, name=load_digits()
            forest = RandomForestClassifier()

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    print(feature_names)

    forest.fit(X_train, y_train)
    y_pred=forest.predict(X_test)
    show_classification_metrics(y_test, y_pred)