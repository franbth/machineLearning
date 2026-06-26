from data.loaders import load_iris, load_wine, load_20newsgroups
from utils.metrics import show_classification_metrics
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

DATASETS = ["iris", "wine", "digits"]

def run(dataset_name):

    match dataset_name:
        case "iris":
            x, y, feature_names, name=load_iris()
            nb = GaussianNB()
        case "wine":
            x, y, feature_names, name=load_wine()     
            nb = GaussianNB()   
        case "20 newsgroups":
            x, y, feature_names, name=load_20newsgroups()
            nb=make_pipeline(TfidfVectorizer(), MultinomialNB())

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)
    show_classification_metrics(y_test, y_pred)