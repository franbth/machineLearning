# TP Integrador – Machine Learning

Aplicación de consola en Python que implementa distintos algoritmos de Machine Learning mediante un menú interactivo. Cada algoritmo se aplica sobre datasets diferentes a los utilizados en los ejemplos de la materia.

Trabajo práctico integrador — Ingeniería en Sistemas, UAI.

## Funcionamiento

1. Se ejecuta `main.py`.
2. Se elige un algoritmo desde el menú.
3. Se elige un dataset entre los disponibles para ese algoritmo.
4. El programa carga los datos, entrena el modelo, muestra las métricas correspondientes y un gráfico.

## Estructura del proyecto

```
machineLearning/
│
├── main.py                  # Menú principal, punto de entrada
│
├── algorithms/
│   ├── __init__.py
│   ├── knn.py
│   ├── linear_regression.py
│   ├── linear_regression_multiple.py
│   ├── polynomial_regression.py
│   ├── svr.py
│   └── decision_tree_regressor.py
│
├── data/
│   ├── __init__.py
│   └── loaders.py           # Carga de datasets (sklearn, OpenML, sintéticos)
│
└── utils/
    ├── __init__.py
    └── metrics.py           # Métricas y gráficos de regresión/clasificación
```

Cada módulo de `algorithms/` declara su propia lista `DATASETS` y una función `run(dataset_name)`, de forma que `main.py` no necesita conocer de antemano qué datasets soporta cada algoritmo.

## Algoritmos implementados

- [x] **Regresión lineal simple** — `algorithms/linear_regression.py`
  Datasets: Diabetes, California Housing, Boston Housing
- [x] **Regresión lineal múltiple** — `algorithms/linear_regression_multiple.py`
  Datasets: California Housing, Wine Quality, Automobile
- [x] **Regresión polinomial** — `algorithms/polynomial_regression.py`
  Datasets: California Housing, Diabetes, Synthetic (`make_regression`)
- [x] **SVR (Support Vector Regression)** — `algorithms/svr.py`
  Datasets: California Housing, Diabetes, Energy Efficiency
- [ ] **Árbol de decisión (regresión)** — `algorithms/decision_tree_regressor.py`
  Datasets: California Housing, Diabetes, Bike Sharing
- [ ] **Random Forest (regresión)**
- [ ] **Regresión logística**
- [x] **KNN** — `algorithms/knn.py`
  Datasets: Iris, Wine, Digits
- [ ] **SVM**
- [ ] **Naive Bayes**
- [ ] **Árbol de decisión (clasificación)**
- [ ] **Random Forest (clasificación)**

## Requisitos

- Python 3.10 o superior (se usa `match/case`)
- scikit-learn
- pandas
- numpy
- matplotlib

Instalación:

```bash
pip install scikit-learn pandas numpy matplotlib
```

## Cómo ejecutar

Ejecutar siempre `main.py`, **no** los archivos dentro de `algorithms/` directamente — los imports son relativos al paquete `machineLearning`, y al correr un submódulo aislado no se resuelven correctamente.

```bash
cd machineLearning
python main.py
```

## Notas técnicas

- Algunos datasets de OpenML requieren especificar `target_column` manualmente, ya que el target por defecto del dataset no siempre coincide con la variable que se busca predecir (ej. `automobile`, `energy-efficiency`).
- Los algoritmos basados en distancias (SVR, KNN, SVM) requieren escalado de variables (`StandardScaler`); los basados en árboles (Decision Tree, Random Forest) no lo necesitan.