import pandas as pd
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import joblib

datos = {
    'historial_crediticio': [1, 0, 1, 0, 1, 0, 1, 1, 0, 1],
    'ingreso': [3000, 1500, 4000, 2000, 3500, 1000, 5000, 4500, 1800, 4200],
    'deuda_actual': [500, 2000, 100, 1500, 300, 2500, 0, 200, 1800, 100],
    'aprobado': [1, 0, 1, 0, 1, 0, 1, 1, 0, 1]
}

df = pd.DataFrame(datos)


X = df[['historial_crediticio', 'ingreso', 'deuda_actual']]
y = df['aprobado']


modelo = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=10,
    random_state=42
)
modelo.fit(X, y)


joblib.dump(modelo, 'modelo_prestamo.pkl')


def predecir_prestamo(historial, ingreso, deuda):
    modelo_cargado = joblib.load('modelo_prestamo.pkl')
    prediccion = modelo_cargado.predict([[historial, ingreso, deuda]])
    return prediccion[0]