
from flask import Flask, request, jsonify
from sklearn.linear_model import LinearRegression
import numpy as np
import spacy

app = Flask(__name__)

# Sample machine learning model 
def train_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

# Sample 3D model generation 
def generate_3d_model(parameters):
    # Your 3D modeling logic here
    # Return the generated 3D model data
    return {"model_data": "Your 3D model data here"}

# Route for training the machine learning model
@app.route('/train_model', methods=['POST'])
def train_model_route():
    data = request.get_json()
    X = data['input_data']
    y = data['output_data']
    model = train_model(X, y)
    return jsonify({"message": "Model trained successfully"})

# Route for generating 3D model
@app.route('/generate_3d_model', methods=['POST'])
def generate_3d_model_route():
    data = request.get_json()
    parameters = data['parameters']
    model_data = generate_3d_model(parameters)
    return jsonify(model_data)

if __name__ == '__main__':
    app.run(debug=True)

app = Flask(__name__)

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def train_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

def generate_3d_model(parameters):
    # Your 3D modeling logic here
    # Return the generated 3D model data
    return {"model_data": "Your 3D model data here"}

# NLP processing function
def process_user_input(user_input):
    doc = nlp(user_input)
    
    # Extract relevant information using spaCy
    extracted_parameters = {
        "parameter1": [token.text for token in doc if token.pos_ == "NOUN"],
        "parameter2": [token.text for token in doc if token.pos_ == "VERB"],
    }
    
    return extracted_parameters

# Route for training the machine learning model
@app.route('/train_model', methods=['POST'])
def train_model_route():
    data = request.get_json()
    X = data['input_data']
    y = data['output_data']
    model = train_model(X, y)
    return jsonify({"message": "Model trained successfully"})

# Route for generating 3D model
@app.route('/generate_3d_model', methods=['POST'])
def generate_3d_model_route():
    data = request.get_json()
    parameters = data['parameters']
    model_data = generate_3d_model(parameters)
    return jsonify(model_data)

# Route for processing user input
@app.route('/process_user_input', methods=['POST'])
def process_user_input_route():
    data = request.get_json()
    user_input = data['userInput']
    extracted_parameters = process_user_input(user_input)
    return jsonify(extracted_parameters)

if __name__ == '__main__':
    app.run(debug=True)

