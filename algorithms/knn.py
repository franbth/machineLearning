from data.loaders import load_iris, load_wine, load_digits
from utils.metrics import show_classification_metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

DATASETS = ["iris", "wine", "digits"]

def run(dataset_name):

    match dataset_name:
        case "iris":
            x, y, feature_names, name=load_iris()
        case "wine":
            x, y, feature_names, name=load_wine()        
        case "digits":
            x, y, feature_names, name=load_digits()

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    knn = KNeighborsClassifier(n_neighbors=5)

    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    show_classification_metrics(y_test, y_pred)
