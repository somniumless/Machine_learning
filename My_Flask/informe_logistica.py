# informe_logistica.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import random

# Función para entrenar el modelo
def entrenar_modelo():
    # Generación de un conjunto de datos más grande (1000 muestras)
    data = {
        'posicion': [random.uniform(0, 100) for _ in range(1000)],  
        'tiros': [random.randint(0, 15) for _ in range(1000)],  
        'tiempo': [random.randint(10, 120) for _ in range(1000)],  
        'resultado': [random.choice([0, 1]) for _ in range(1000)]  
    }
    
    df = pd.DataFrame(data)
    
    # División de las variables independientes (X) y la variable dependiente (y)
    X = df[['posicion', 'tiros', 'tiempo']]
    y = df['resultado']
    
    # División del conjunto de datos en entrenamiento (80%) y prueba (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Crear y entrenar el modelo de Regresión Logística
    modelo = LogisticRegression(max_iter=1000)
    modelo.fit(X_train, y_train)
    
    # Realizar las predicciones en el conjunto de prueba
    y_pred = modelo.predict(X_test)
    
    # Calcular las métricas de evaluación
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"Matriz de Confusión:\n{conf_matrix}")
    
    return modelo

# Función para predecir el resultado
def predecir_resultado(modelo, posicion, tiros, tiempo):
    """
    Predice el resultado de un partido en función de los parámetros de entrada:
    - posicion: posición del balón en el campo (%).
    - tiros: número de tiros al arco.
    - tiempo: tiempo de juego en minutos.
    
    El modelo devuelve 'Ganar' o 'Perder' en función de la predicción.
    """
  
    datos = [[posicion, tiros, tiempo]]
    

    prediccion = modelo.predict(datos)
    
  
    if prediccion[0] == 1:
        return "Ganar"
    else:
        return "Perder"



