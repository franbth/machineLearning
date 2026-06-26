import algorithms.knn as knn
import algorithms.linear_regression_simple as linear_regression_simple
import algorithms.linear_regression_multiple as linear_regression_multiple
import algorithms.polynomial_regression as polynomial_regression
import algorithms.svr as svr
import algorithms.decision_tree_regressor as decision_tree_regressor
import algorithms.random_forest_regressor as random_forest_regressor
import algorithms.logistic_regressor as logistic_regressor
import algorithms.svm as svm
import algorithms.naive_bayes as naive_bayes
import algorithms.decission_tree_classifier as decission_tree_classifier
import algorithms.random_forest_classifier as random_forest_classifier
from utils.animation import show_welcome


ALGORITHMS = [
    ("KNN", knn),
    ("Regresión lineal simple", linear_regression_simple),
    ("Regresión lineal múltiple", linear_regression_multiple),
    ("Regresión polinomial", polynomial_regression),
    ("SVR", svr),
    ("Árbol de decisión (regresión)", decision_tree_regressor),
    ("Random Forest (regresión)", random_forest_regressor),
    ("Regresión logística", logistic_regressor),
    ("SVM", svm),
    ("Naive Bayes", naive_bayes),
    ("Árbol de decisión (clasificación)", decission_tree_classifier),
    ("Random Forest (clasificación)", random_forest_classifier),
]

def read_int(prompt, min_value, max_value):
    while True:
        raw_value = input(prompt).strip()

        try:
            value = int(raw_value)
        except ValueError:
            print("Ingresá un número válido.")
            continue

        if value < min_value or value > max_value:
            print(f"Ingresá un número entre {min_value} y {max_value}.")
            continue

        return value


def show_algorithms():
    print("\n=== Menú principal ===")
    for index, (label, _) in enumerate(ALGORITHMS, start=1):
        print(f"{index}. {label}")
    print("0. Salir")


def show_datasets(algorithm_label, algorithm_module):
    datasets = getattr(algorithm_module, "DATASETS", [])

    while True:
        print(f"\n=== {algorithm_label} ===")
        for index, dataset in enumerate(datasets, start=1):
            print(f"{index}. {dataset}")
        print("0. Volver")

        choice = read_int("Elegí un dataset: ", 0, len(datasets))
        if choice == 0:
            return

        dataset_name = datasets[choice - 1]
        algorithm_module.run(dataset_name)
        input("\nPresioná Enter para volver al menú principal...")
        return


def main():
    show_welcome()
    while True:
        show_algorithms()
        choice = read_int("Elegí un algoritmo: ", 0, len(ALGORITHMS))

        if choice == 0:
            print("Saliendo...")
            break

        algorithm_label, algorithm_module = ALGORITHMS[choice - 1]
        show_datasets(algorithm_label, algorithm_module)


if __name__ == "__main__":
    main()
