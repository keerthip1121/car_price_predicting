# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from flask import Flask,request,render_template
import numpy as np
import pandas as pd
import joblib


# initialise flask
app = Flask(__name__,template_folder='template')

#load model
model = joblib.load('car_catboost.pkl')

# launch home page
@app.route('/',methods = ['GET'])
def home():
    # load html page
    return render_template('test.html')
@app.route('/',methods = ['POST'])
def prediction():
    x_col = ['Name','Owner','Month_of_Purchase','Year_Of_Purchase',
             'Kilometers','Rating','Fuel_Type','Transmission','RTO',
             'Insurance_Type','Insurance']
    # info from user input
    
    d =[[x for x in request.form.values()]]
    
    data = pd.DataFrame(d,columns=x_col)
    data = data[['Name','Rating','Kilometers','Month_of_Purchase',
             'Year_Of_Purchase','Owner','Fuel_Type','Transmission','RTO',
             'Insurance','Insurance_Type']]
    prediction = model.predict(data)[0]
    print(prediction)
    
    text = "₹Prediction "+str(np.round(prediction))
    print(text)
    return render_template('test.html',prediction = text)
if __name__ == '__main__':
    app.run(debug=True)
