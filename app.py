from flask import Flask, render_template, redirect, url_for, request, session, jsonify
import json
import requests
import os
from PIL import Image
import io
import numpy as np
import tensorflow as tf
from google.generativeai import GenerativeModel, configure
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch


app = Flask(__name__)
app.secret_key = 'your_secret_key'

configure(api_key="AIzaSyD3CiD16W3vsYD3Og144jtfbyIhjxlYiLU")

# Create the model
generation_config = {
    "temperature": 1.9,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = GenerativeModel(
    model_name="gemini-1.5-pro",
    generation_config=generation_config,
    system_instruction="""
    Your are Groot. You will only respond to questions related to gardening. 
    Keep your answers concise, with a maximum of 50 words. 
    Remove any bold formatting from your replies. 
    If your answer contains multiple points, display each point on a new line.
    """
)

chat_session = model.start_chat(
    history=[]
)


# Load user data from JSON file
# paste the users.json path in the below function
with open(r'users.json', 'r') as f:
    users = json.load(f)

# Load plant data and ML model
#plant_data = json.load(open('plant_data.json'))
#model = tf.keras.models.load_model('plant_disease_model.h5')

# Weather API endpoint
weather_api_url = 'http://api.openweathermap.org/data/2.5/weather?q={}&appid={}&units=metric'
weather_api_key = 'ad16a027f6714488597ae31134fe887b'

@app.route('/')
def index():
    if 'username' in session:
        city = session.get('city', 'Default City')  # Provide a default value if city is not set
        weather = get_weather(city)  # Call get_weather with the city
        return render_template('index.html', username=session['username'], weather=weather)
    else:
        return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users and users[username]['password'] == password:
            session['username'] = username
            session['city'] = users[username]['city']
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error='Invalid username or password')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        city = request.form['city']
        if username not in users:
            users[username] = {'password': password, 'city': city}
            with open('users.json', 'w') as f:
                json.dump(users, f)
            return redirect(url_for('login'))
        else:
            return render_template('register.html', error='Username already exists')
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('username', None)  # Remove username from session
    return redirect(url_for('login'))  # Redirect to login page

@app.route('/plant_suggestion', methods=['GET', 'POST'])
def plant_suggestion():
    city = session.get('city', 'Default City')  # Provide a default value if city is not set
    weather = get_weather(city)  # Get weather information for the city

    if request.method == 'POST':
        plant_type = request.form['plant_type']  # Get the plant type from the form
        suggestion = suggest_plant(plant_type, weather)  # Get the plant suggestion based on the weather
        return render_template('plant_suggestion.html', suggestion=suggestion, weather=weather)

    # For GET requests, render the template with the weather information
    return render_template('plant_suggestion.html', weather=weather)

@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot():
    if request.method == 'POST':
        user_message = request.json.get('message')
        response = chat_session.send_message(user_message)

        # Process the response to handle point-wise output
        response_text = response.text
        # Split the response into lines
        response_lines = response_text.split('\n')
        # Join the lines with HTML line breaks
        formatted_response = "<br>".join(response_lines)

        return jsonify({'response': formatted_response})
    
    return render_template('chatbot.html')

processor = AutoImageProcessor.from_pretrained("SanketJadhav/PlantDiseaseClassifier-Resnet50")
model = AutoModelForImageClassification.from_pretrained("SanketJadhav/PlantDiseaseClassifier-Resnet50")

@app.route('/disease_detection', methods=['GET', 'POST'])
def disease_detection():
    city = session.get('city', 'Default City')  # Provide a default value if city is not set
    weather = get_weather(city)
    disease = None  # Initialize disease variable
    if request.method == 'POST':
        # Check if the 'image' key is in the request.files
        if 'image' not in request.files:
            return "No file part", 400  # Handle the case where no file is uploaded

        file = request.files['image']
        if file.filename == '':
            return "No selected file", 400  # Handle the case where no file is selected

        # Read and process the uploaded image
        img = Image.open(io.BytesIO(file.read()))

        # Preprocess the image using the processor
        inputs = processor(images=img, return_tensors="pt")

        # Make a prediction
        with torch.no_grad():
            outputs = model(**inputs)

        # Get the predicted class index
        predicted_class_idx = torch.argmax(outputs.logits, dim=1).item()

        # Get the class labels (assuming the model has a 'id2label' attribute)
        predicted_label = model.config.id2label[predicted_class_idx]
        disease = predicted_label  # Set the detected disease

        return render_template('disease_detection.html', disease=disease, weather=weather)  # Render the template with the detected disease

    return render_template('disease_detection.html',  weather=weather)  # Render the template for GET requests

import requests

def get_weather(city):
    # Make the API request to get the weather data
    response = requests.get(weather_api_url.format(city, weather_api_key))
    
    # Check if the response is successful
    if response.status_code != 200:
        return {
            'temperature': None,
            'description': 'Unable to retrieve weather data',
            'emoji': '❓'  # Default emoji for error
        }

    # Parse the JSON response
    data = response.json()

    # Extract temperature and weather description
    temperature = data['main']['temp']
    description = data['weather'][0]['description']

    # Determine the emoji based on the weather description
    if "clear" in description:
        emoji = "☀️"  # Sun emoji for clear weather
    elif "cloud" in description:
        emoji = "☁️"  # Cloud emoji for cloudy weather
    elif "rain" in description or "drizzle" in description:
        emoji = "🌧️"  # Rain emoji for rainy weather
    elif "snow" in description:
        emoji = "❄️"  # Snowflake emoji for snowy weather
    elif "storm" in description:
        emoji = "⛈️"  # Storm emoji for stormy weather
    else:
        emoji = "⛅"  # Rainbow emoji for other conditions

    # Return the weather data with emoji
    return {
        'temperature': temperature,
        'description': description,
        'emoji': emoji
    }

def load_plant_data():
    # paste the plant_data.json path in the below function
    with open(r'plant_data.json', 'r') as json_file:
        return json.load(json_file)

def suggest_plant(plant_type, weather):
    # Get the temperature and description from the weather data
    temperature = weather['temperature']
    description = weather['description']
    plant_data = load_plant_data()
    plant_type=plant_type.lower()
    # Check if the plant type exists in the plant data
    if plant_type in plant_data:
        ideal_conditions = plant_data[plant_type]['ideal_conditions']
        
        # Check if the current temperature and weather conditions match the ideal conditions
        if (ideal_conditions['temperature'][0] <= temperature <= ideal_conditions['temperature'][1] and 
                description in ideal_conditions['weather']):
            return plant_data[plant_type]['suggestion']
        else:
            return plant_data[plant_type]['alternative']
    else:
        return "The plant type '{}' is not recognized. Please provide a valid plant type.".format(plant_type)
if __name__ == '__main__':
    app.run(debug=True)
