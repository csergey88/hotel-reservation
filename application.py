import joblib
import os
import numpy as np
import pandas as pd
from config.paths_config import MODEL_OUTPUT_PATH
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

loaded_model = joblib.load(MODEL_OUTPUT_PATH)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            def get_field(name, cast):
                value = request.form.get(name)
                if value is None or value == "":
                    raise ValueError(f"Missing field: {name}")
                try:
                    return cast(value)
                except ValueError:
                    raise ValueError(f"Invalid value for {name}: {value}")

            lead_time = get_field('lead_time', float)
            no_of_special_requests = get_field('no_of_special_requests', float)
            avg_price_per_room = get_field('avg_price_per_room', float)
            arrival_month = get_field('arrival_month', int)
            arrival_date = get_field('arrival_date', int)
            market_segment_type = get_field('market_segment_type', int)
            no_of_week_nights = get_field('no_of_week_nights', int)
            no_of_weekend_nights = get_field('no_of_weekend_nights', int)
            type_of_meal_plan = get_field('type_of_meal_plan', int)
            room_type_reserved = get_field('room_type_reserved', int)
            

            features = np.array([[lead_time, no_of_special_requests, avg_price_per_room,
                                  arrival_month, arrival_date, market_segment_type,
                                  no_of_week_nights, no_of_weekend_nights,
                                  type_of_meal_plan, room_type_reserved]])

            prediction = loaded_model.predict(features)

            return render_template('index.html', prediction=prediction[0], error=None)
        except Exception as e:
            return render_template('index.html', error=str(e), prediction=None)
    return render_template('index.html', prediction=None)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)
