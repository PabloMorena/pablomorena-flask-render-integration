from flask import Flask, request, render_template
from pickle import load

app = Flask(__name__)

# Cargar el modelo entrenado
model = load(open("/workspaces/pablomorena-machine-learning-python-templat-ML-WEB-APP-USING-FLASK/src/KNeighbors.sav", "rb"))

# Diccionario para traducir predicciones a etiquetas más legibles
class_dict = {
    "3": "Calidad 3",
    "4": "Calidad 4",
    "5": "Calidad 5",
    "6": "Calidad 6",
    "7": "Calidad 7",
    "8": "Calidad 8",
}

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            # Recoger valores del formulario
            inputs = [
                float(request.form["fixed_acidity"]),
                float(request.form["volatile_acidity"]),
                float(request.form["citric_acid"]),
                float(request.form["residual_sugar"]),
                float(request.form["chlorides"]),
                float(request.form["free_sulfur_dioxide"]),
                float(request.form["total_sulfur_dioxide"]),
                float(request.form["density"]),
                float(request.form["pH"]),
                float(request.form["sulphates"]),
                float(request.form["alcohol"]),
            ]

            # Hacer predicción
            prediction = str(model.predict([inputs])[0])
            pred_class = class_dict.get(prediction, f"Clase desconocida: {prediction}")

        except Exception as e:
            pred_class = f"Error en la predicción: {str(e)}"
    else:
        pred_class = None

    return render_template("index.html", prediction=pred_class)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")