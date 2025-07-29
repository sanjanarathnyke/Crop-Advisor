# Crop Advisor

A web application that recommends the most suitable crop to grow based on soil and environmental parameters using a machine learning model.

## Features
- User-friendly web interface (Flask)
- Predicts the best crop based on:
  - Nitrogen (N)
  - Phosphorus (P)
  - Potassium (K)
  - Temperature
  - pH
  - Rainfall
- Trained ML model included (`trained_model.pkl`)

---

## How to Run the Application

### 1. Clone the Repository
```
git clone <repo-url>
cd Crop-Advisor
```

### 2. Install Dependencies
Make sure you have Python 3.x installed. Then install the required packages:
```
pip install flask numpy scikit-learn
```

### 3. Run the Application
```
python app.py
```
The app will be available at [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## ML Model Training Steps

1. **Collect Data**
   - Use a dataset containing soil and climate features (N, P, K, temperature, pH, rainfall) and the target crop label.

2. **Preprocess Data**
   - Handle missing values, if any.
   - Encode categorical labels (e.g., using `LabelEncoder` from scikit-learn).
   - Split data into training and testing sets.

3. **Train the Model**
   - Example using RandomForestClassifier:
     ```python
     from sklearn.ensemble import RandomForestClassifier
     from sklearn.preprocessing import LabelEncoder
     import pandas as pd
     import pickle

     # Load your dataset
     data = pd.read_csv('your_dataset.csv')
     X = data[['N', 'P', 'K', 'temperature', 'ph', 'rainfall']]
     y = data['label']

     le = LabelEncoder()
     y_encoded = le.fit_transform(y)

     model = RandomForestClassifier()
     model.fit(X, y_encoded)

     # Save the model and label encoder
     with open('trained_model.pkl', 'wb') as f:
         pickle.dump((model, le), f)
     ```

4. **Test the Model**
   - Evaluate accuracy and performance on the test set.

5. **Integrate with Flask App**
   - Place the `trained_model.pkl` file in the project directory.
   - The Flask app will load and use this model for predictions.

---

## File Structure
```
app.py                  # Flask web app
trained_model.pkl       # Trained ML model and label encoder
crop_Recomendation_system.ipynb  # (Optional) Model training notebook
templates/
    index.html          # Web interface
```

---
