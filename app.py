import numpy as np
from flask import Flask, request, render_template, jsonify
import pickle
import load
import os

# GLOBAL VARIABLES
SECRET_KEY = os.urandom(32)
PATH = os.getcwd()
N_CLASSES = 4

model_path = PATH + '/model/model_RF.pkl'

# Apps init
app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config['SECRET_KEY'] = SECRET_KEY


@app.route("/")
def home():
    return render_template('index.html')

@app.route("/prediction/", methods = ['GET', 'POST'])
def predict():
    if request.method == "POST":
 # Input data dari Base64
        Age = request.form.get('Age')
        BMI = request.form.get('BMI')
        Glucose = request.form.get('Glucose')
        Insulin = request.form.get('Insulin')
        HOMA = request.form.get('HOMA')
        Leptin = request.form.get('Leptin')
        Adiponectin = request.form.get('Adiponectin')
        Resistin = request.form.get('Resistin')
        MCP1 = request.form.get('MCP.1')
        
        model = load(open(PATH + '/model/model_RF.pkl','rb'))
        row = [[float(Age), float(BMI), 
        float(Glucose), float(Insulin), 
        float(HOMA),
        float(Leptin), float(Adiponectin), float(Resistin), float(MCP1)]]

        input_data_as_numpy_array = np.asarray(row)
        # input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)      
        yhat = model.predict(input_data_as_numpy_array)
        print('Prediction: %d' % yhat[0])
        if yhat[0] == 1:
            label = "Dari hasil tes kanker payudara yang diinput, teridentifikasi kanker payudara"
        elif yhat[0] == 2:
            label = "Dari hasil tes kanker payudara yang diinput, tidak teridentifikasi kanker payudara"
        return jsonify({'status' : 'Berhasil','data' : label})
   

if __name__ =="__main__":
    app.run(debug=True)
