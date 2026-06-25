from data.loaders import load_breast_cancer, load_titanic, load_heart_disease
from utils.metrics import show_classification_metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


DATASETS = ["breast cancer", "titanic", "heart disease"]

def run(dataset_name):

    match dataset_name:
        case "breast cancer":
            x, y, feature_names, name=load_breast_cancer()
        case "titanic":
            x, y, feature_names, name=load_titanic()        
        case "heart disease":
            x, y, feature_names, name=load_heart_disease()
    print (type(x))
    print(type(y))

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    logReg = LogisticRegression(max_iter=500) #scikit learn pide mas iteraciones pero da los mismos resultados con 100
    logReg.fit(X_train, y_train)
    y_predict = logReg.predict(X_test)
    show_classification_metrics(y_test, y_predict)


