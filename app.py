from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

# Load model and encoders
model = joblib.load('AI/exoplanet_model.pkl')
label_encoder = joblib.load('AI/label_encoder.pkl')
feature_columns = joblib.load('AI/feature_columns.pkl')
feature_encoders = joblib.load('AI/feature_encoders.pkl')
imputer_numeric = joblib.load('AI/imputer_numeric.pkl')
imputer_cat = joblib.load('AI/imputer_cat.pkl')
numeric_cols = joblib.load('AI/numeric_columns.pkl')
cat_cols = joblib.load('AI/cat_columns.pkl')
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = request.json
        
        # Convert to DataFrame
        df_new = pd.DataFrame([data])
        
        # Impute numeric columns
        numeric_cols_exist = [c for c in numeric_cols if c in df_new.columns]
        if numeric_cols_exist:
            df_new[numeric_cols_exist] = imputer_numeric.transform(df_new[numeric_cols_exist])
        
        # Impute categorical columns
        cat_cols_exist = [c for c in cat_cols if c in df_new.columns]
        if cat_cols_exist:
            df_new[cat_cols_exist] = imputer_cat.transform(df_new[cat_cols_exist])
        
        # Encode categorical features
        for col, encoder in feature_encoders.items():
            if col in df_new.columns:
                df_new[col] = df_new[col].apply(lambda x: x if x in encoder.classes_ else encoder.classes_[0])
                df_new[col] = encoder.transform(df_new[col])
        
        # Ensure correct column order
        df_new = df_new[feature_columns]
        
        # Make prediction
        prediction = model.predict(df_new)
        proba = model.predict_proba(df_new)
        
        # Prepare response
        result = {
            'prediction': label_encoder.inverse_transform(prediction)[0],
            'probabilities': {
                cls: float(prob) for cls, prob in zip(label_encoder.classes_, proba[0])
            }
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)