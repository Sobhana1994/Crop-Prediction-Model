from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import requests
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Requests

# Load and Process Dataset
try:
    df1 = pd.read_csv("Crop_recommendation.csv")
    df2 = pd.read_csv("CRUSP.csv")
except FileNotFoundError:
    print("Error: Dataset files not found!")
    exit()

# Encode Crop Labels
all_crops = list(set(df1['label'].unique()).union(set(df2['label'].unique())))
label_encoder = LabelEncoder()
label_encoder.fit(all_crops)

df1['label'] = label_encoder.transform(df1['label'])
df2['label'] = label_encoder.transform(df2['label'])

df_combined = pd.concat([df1, df2], ignore_index=True)

# Identify Columns
categorical_cols = ['Soilcolor'] if 'Soilcolor' in df_combined.columns else []
numerical_cols = [col for col in df_combined.columns if col not in ['label'] + categorical_cols]

# Fill Missing Values
df_combined[numerical_cols] = df_combined[numerical_cols].apply(lambda x: x.fillna(x.mean()))
df_combined[categorical_cols] = df_combined[categorical_cols].apply(lambda x: x.fillna(x.mode()[0]))

# Encode Categorical Data
for col in categorical_cols:
    df_combined[col] = LabelEncoder().fit_transform(df_combined[col].astype(str))

# Standardize Data
scaler = StandardScaler()
df_combined[numerical_cols] = scaler.fit_transform(df_combined[numerical_cols])

# Train Model
X = df_combined[numerical_cols + categorical_cols]
y = df_combined['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=300, max_depth=25, random_state=42)
model.fit(X_train, y_train)
print(f"Model Accuracy: {accuracy_score(y_test, model.predict(X_test)):.2%}")

# API Keys
WEATHER_API_KEY = "ed3f7c277617620ec97ffe73b5362ce7"

# Get Coordinates from Location
def get_coordinates(location):
    url = f"http://api.openweathermap.org/geo/1.0/direct?q={location}&limit=1&appid={WEATHER_API_KEY}"
    try:
        response = requests.get(url)
        data = response.json()
        if data:
            return data[0]['lat'], data[0]['lon']
    except:
        return None, None
    return None, None

# Get Weather Data
def get_weather_data(lat, lon):
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}&units=metric"
    try:
        response = requests.get(url)
        data = response.json()
        if 'main' in data:
            return data['main'].get('temp', None), data['main'].get('humidity', None), data.get('rain', {}).get('1h', 0)
    except:
        return None, None, None
    return None, None, None

# Get Soil Data
def get_soil_data(lat, lon):
    url = f"https://rest.isric.org/soilgrids/v2.0/properties/query?lon={lon}&lat={lat}&properties=phh2o,ocd,nitrogen"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            soil_props = data.get('properties', {}).get('layers', [])
            if soil_props:
                return {
                    "pH": soil_props[0]['depths'][0]['values'].get('phh2o', 6.5),
                    "OrganicCarbon": soil_props[0]['depths'][0]['values'].get('ocd', 1.2),
                    "Nitrogen": soil_props[0]['depths'][0]['values'].get('nitrogen', 0.15)
                }
    except:
        pass
    return {"pH": 6.5, "OrganicCarbon": 1.2, "Nitrogen": 0.15}

# Predict Crops
def predict_top_crops(lat, lon, top_n=10):
    temp, humidity, rainfall = get_weather_data(lat, lon)
    soil_data = get_soil_data(lat, lon)

    if temp is None or soil_data is None:
        return ["Could not retrieve weather or soil data"]

    input_data = {**soil_data, 'temperature': temp, 'humidity': humidity, 'rainfall': rainfall}
    input_df = pd.DataFrame([input_data], columns=numerical_cols + categorical_cols)
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

    probabilities = model.predict_proba(input_df)[0]
    crop_labels = label_encoder.inverse_transform(model.classes_)
    top_indices = probabilities.argsort()[-top_n:][::-1]
    top_crops = [(crop_labels[i], probabilities[i]) for i in top_indices if crop_labels[i] in all_crops]

    return [f"{crop} (Confidence: {prob:.2%})" for crop, prob in top_crops]

# API Endpoint
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    location = data.get("location", "").strip()

    if not location:
        return jsonify({"error": "Location is required"}), 400

    lat, lon = get_coordinates(location)
    if lat is None:
        return jsonify({"error": "Invalid location"}), 400

    crops = predict_top_crops(lat, lon)
    return jsonify({"prediction": crops})

# Run Flask App
if __name__ == '__main__':
    app.run(debug=True)
