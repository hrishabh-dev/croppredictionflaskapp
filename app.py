from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import numpy as np
import pickle
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

flask_app = Flask(__name__)
CORS(flask_app)
model = pickle.load(open("model.pkl", "rb"))

@flask_app.route("/") 
def home():
    return render_template("index.html")

@flask_app.route("/predict", methods=["POST"])
def predict():
    if request.is_json:
        data = request.get_json()
        float_features = [
            data['N'],
            data['P'],
            data['K'],
            data['temperature'],
            data['humidity'],
            data['pH'], 
            data['rainfall']
        ]
    else: 
        try:
            float_features = [
                float(request.form['Nitrogen']),
                float(request.form['Phosphorus']),
                float(request.form['Potassium']),
                float(request.form['temperature']),
                float(request.form['humidity']),
                float(request.form['pH']), 
                float(request.form['rainfall'])
            ]
        except KeyError as e:
            return render_template("index.html", prediction_text=f"Error: Missing form field - {e}. Please fill all fields.")
        except ValueError:
            return render_template("index.html", prediction_text="Error: Please enter valid numbers for all fields.")


    features = [np.array(float_features)]
    prediction = model.predict(features)
    if request.is_json:
        return jsonify({"prediction": prediction[0]})
    else:
        return render_template("index.html", prediction_text="The Predicted Crop is {}".format(prediction[0])) # Use prediction[0] here too

if __name__ == "__main__":
    flask_app.run(debug=True)
