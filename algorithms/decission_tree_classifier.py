from data.loaders import load_iris, load_breast_cancer, load_digits
from utils.metrics import show_classification_metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

DATASETS = ["iris", "breast cancer",  "digits"]

def run(dataset_name):
    match dataset_name:
        case "iris":
            x, y, feature_names, name=load_iris() 
            tree = DecisionTreeClassifier() 
        case "breast cancer":
            x, y, feature_names, name=load_breast_cancer() 
            tree = DecisionTreeClassifier() #Sin max_depth tiene sobreajuste
        case "digits":
            x, y, feature_names, name=load_digits()
            tree = DecisionTreeClassifier()

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    print(feature_names)

    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    show_classification_metrics(y_test, y_pred)