from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

with open('trained_model.pkl', 'rb') as f:
    model, le = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Read inputs from form
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        temperature = float(request.form['temperature'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        # Prepare input for prediction
        input_data = np.array([[N, P, K, temperature, ph, rainfall]])
        predicted_label = model.predict(input_data)[0]
        # Decode the predicted label to crop name
        predicted_crop = le.inverse_transform([predicted_label])[0]

        return render_template('index.html', result=f"Recommended Crop: {predicted_crop}")
    except Exception as e:
        return render_template('index.html', result=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)