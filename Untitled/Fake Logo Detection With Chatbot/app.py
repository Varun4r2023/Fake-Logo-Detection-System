from flask import Flask, render_template, request
import cv2
from tensorflow.keras.models import load_model
import numpy as np
import base64
import os

app = Flask(__name__)

# Load the model without compilation to avoid deserialization issues
model_path = os.path.join("model", "model.h5")
new_model = load_model(model_path, compile=False)

# Function to process uploaded image
def process_image(image):
    new_height, new_width = 256, 256
    resized_image = cv2.resize(image, (new_width, new_height))  # Resize image
    normalized_image = resized_image / 255.0  # Normalize pixel values
    pred = new_model.predict(np.expand_dims(normalized_image, axis=0))  # Predict
    return pred[0][0]  # Return prediction confidence

# Function to convert image to base64
def image_to_base64(image):
    _, img_encoded = cv2.imencode('.jpg', image)
    img_base64 = base64.b64encode(img_encoded).decode('utf-8')
    return img_base64

# Home page
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

# Result page
@app.route('/result', methods=['POST'])
def result():
    if 'file' not in request.files:
        return render_template('index.html', message='No file uploaded')

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', message='No file selected')

    if file:
        # Read and decode image
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

        if image is None:
            return render_template('index.html', message='Invalid image file')

        # Process image
        pred = process_image(image)
        img_base64 = image_to_base64(image)

        # Determine result
        if pred > 0.5:
            result_text = "Provided Logo is Fake"
        else:
            result_text = "Provided Logo is Real"

        return render_template('result.html', prediction=result_text, image=img_base64)

if __name__ == '__main__':
    app.run(debug=True)
