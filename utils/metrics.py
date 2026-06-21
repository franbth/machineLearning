import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import numpy as np


def show_regression_metrics(y_real, y_pred):
    mse = metrics.mean_squared_error(y_real, y_pred)
    mae = metrics.mean_absolute_error(y_real, y_pred)
    r2 = metrics.r2_score(y_real, y_pred)
    rmse = np.sqrt(mse)

    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"R^2 Score: {r2:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")

    plt.scatter(y_real, y_pred, alpha=0.5)
    plt.xlabel("Actual Values") 
    plt.ylabel("Predicted Values")
    plt.title("Actual vs Predicted Values") 
    plt.plot([y_real.min(), y_real.max()], 
         [y_real.min(), y_real.max()], 
         'r--')  # línea roja punteada
    plt.show()

def show_classification_metrics(y_real, y_pred):
    accuracy = metrics.accuracy_score(y_real, y_pred)
    precision = metrics.precision_score(y_real, y_pred, average='weighted')
    recall = metrics.recall_score(y_real, y_pred, average='weighted')
    f1 = metrics.f1_score(y_real, y_pred, average='weighted')

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    metrics.ConfusionMatrixDisplay.from_predictions(y_real, y_pred)
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()
