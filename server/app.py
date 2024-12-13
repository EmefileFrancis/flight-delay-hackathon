from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('../flight_delay_model.pkl')

# Load the data from the CSV file
data = pd.read_csv('../data/airports.csv')

@app.route('/airports', methods=['GET'])
def get_airports():
    # Select relevant columns
    airports = data[['airport_id', 'airport_name']]
    
    # Drop duplicates
    airports = airports.drop_duplicates()
    
    # Sort by airport name
    airports = airports.sort_values(by='airport_name    git reset --soft HEAD~1')
    
    # Convert to list of dictionaries
    airports_list = airports.to_dict(orient='records')
    
    return jsonify(airports_list)


@app.route('/predict_delay', methods=['POST'])
def predict_delay():
    # Get the JSON data from the request
    data = request.get_json()
    
    # Extract the parameters
    day_of_week = data['day_of_week']
    origin_airport_id = data['origin_airport_id']
    dest_airport_id = data['dest_airport_id']
    
    # Prepare the input data
    input_data = np.array([[day_of_week, origin_airport_id, dest_airport_id]])
    
    # Make prediction
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]
    
    # Prepare the response
    response = {
        'prediction': int(prediction[0]),
        'confidence': float(probability)
    }
    
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)