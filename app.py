import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import joblib
from sklearn.preprocessing import OneHotEncoder

app = Flask(__name__)  # Initialize the flask App
model = joblib.load(open("Honda.pkl", "rb"))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        Car_Model=[0,0,0,0,0]
        Car_Model[int(request.form['Car Model'])]=1
        Fuel_Type_Petrol = int(request.form['Fuel_Type_Petrol'])
        Year = int(request.form['Year'])
        Kms_Driven=int(request.form['Kms_Driven'])
        Transmission_Manual=int(request.form['Transmission_auto'])
        input_data=[Year,Kms_Driven,Fuel_Type_Petrol,Transmission_Manual]+Car_Model

        prediction=model.predict([input_data])
    return render_template('index.html', prediction_text='Estimated Price: '+str(round(float(prediction[0]),2)))


if __name__ == "__main__":
    app.run(debug=True)