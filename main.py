import os
from flask import Flask, request, jsonify
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib

app = Flask(__name__)

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess the data (scaling)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Save the trained model
joblib.dump(model, 'iris_model.pkl')

# Define a route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        data = request.json
        features = [data['sepal_length'], data['sepal_width'], data['petal_length'], data['petal_width']]
        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)[0]
        return jsonify({'predicted_class': iris.target_names[prediction]})
    return jsonify({'error': 'Invalid request method'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
