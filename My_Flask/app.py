from flask import Flask, render_template, request
from datetime import datetime
import re 
import os
import Linear_Regression_model
from Linear_Regression_model import calculateGrade
from informe_logistica import predecir_resultado, entrenar_modelo


app = Flask(__name__)

@app.route("/")
def home():
    return "Hello, flask!"

@app.route("/hello/<name>")
def hello_there(name):
    now = datetime.now()

    match_object = re.match("[a-zA-Z]+", name)

    if match_object:
        clean_name = match_object.group(0)
    else:
        clean_name = "Friend"

    content = "Hello there, " + clean_name + "! Hour: " + str(now)
    return content

@app.route("/pagina")
def pagina():
    return render_template("pagina.html")


@app.route("/logistica")
def logistica():
    return render_template("logistica.html")

@app.route('/regresion', methods=['GET', 'POST'])
def linear_regression():
    result = None
    if request.method == 'POST':
        try:
            hours = float(request.form['hours'])
            result = round(calculateGrade(hours), 2)
        except ValueError:
            result = "Entrada inválida."
    return render_template('LinearRegressionGrades.html', result=result)

modelo = entrenar_modelo()


@app.route("/predecir", methods=['POST', 'GET'])
def predecir():
    resultado = None
    if request.method == 'POST':
        try:
            # Obtener los datos del formulario
            posicion = float(request.form['posicion'])
            tiros = int(request.form['tiros'])
            tiempo = int(request.form['tiempo'])

            # Llamar a la función para predecir el resultado
            resultado = predecir_resultado(modelo, posicion, tiros, tiempo)
        except (ValueError, KeyError):
            return "Error en los datos. Asegúrate de que los campos estén completos y sean válidos.", 400
    
    return render_template('informe_logistica.html', resultado=resultado)

