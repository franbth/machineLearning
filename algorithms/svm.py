from data.loaders import load_breast_cancer, load_iris, load_digits
from utils.metrics import show_classification_metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

DATASETS = ["breast cancer", "iris", "digits"]
KERNELS = ['linear', 'poly', 'rbf', 'sigmoid']

def run(dataset_name):

    match dataset_name:
        case "breast cancer":
            x, y, feature_names, name=load_breast_cancer() 
            svm= SVC(kernel='rbf')
        case "iris":
            x, y, feature_names, name=load_iris()
            svm = SVC()
        case "digits":
            x, y, feature_names, name=load_digits()
            svm= SVC()

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    print(feature_names)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)  # ojo: sin fit acá

    svm.fit(X_train_scaled, y_train)
    y_pred=svm.predict(X_test_scaled)
    show_classification_metrics(y_test, y_pred)