from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import numpy as np
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

app=Flask(__name__)
model=joblib.load('employee_attrition_model.pkl')#MODEL IMPORT

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data=request.get_json()#getting data from frontend
        df=pd.DataFrame([data])#FRONTEND(data)->DATAFRAME
        numeric_cols = [
            "Age",
            "JobInvolvement",
            "NumCompaniesWorked",
            "StockOptionLevel",
            "JobSatisfaction",
            "YearsSinceLastPromotion",
            "EnvironmentSatisfaction",
            "TotalWorkingYears",
            "DistanceFromHome"
        ]
        for col in numeric_cols:
            df[col]=pd.to_numeric(df[col])
        prediction=model.predict(df)[0]
        probability=model.predict_proba(df)[0]
        confidence=round(max(probability)*100,2)
        result= 'Yes' if prediction==1 else 'No'
        return jsonify({
            'prediction':result,
            'confidence':confidence
        })
    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 400 

if __name__=='__main__':
    app.run(debug=True)