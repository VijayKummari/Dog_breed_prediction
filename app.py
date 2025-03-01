
###Previoussss sample code
'''

from flask import Flask, render_template, request, redirect, url_for
import requests

app = Flask(__name__)

API_KEY = "live_lIstnMtl9oUlFnN1a7G7z4bv1P8FOA40BKO2QBi0biydIzmDitDCsfyhQwbLpUwA"  # Replace with your API key from The Dog API

def get_breed_info(breed):
    url = f"https://api.thedogapi.com/v1/breeds/search?q={breed}"
    headers = {"x-api-key": API_KEY}
    response = requests.get(url, headers=headers)
    if response.status_code == 200 and response.json():
        data = response.json()[0]  # get the first match
        info = {
            "name": data.get("name", "Unknown"),
            "bred_for": data.get("bred_for", "Unknown"),
            "temperament": data.get("temperament", "Unknown"),
            "life_span": data.get("life_span", "Unknown"),
            "origin": data.get("origin", "Unknown"),
            "weight": data.get("weight", {}).get("metric", "Unknown") + " kg" if data.get("weight") else "Unknown",
            "height": data.get("height", {}).get("metric", "Unknown") + " cm" if data.get("height") else "Unknown",
            "breed_group": data.get("breed_group", "Unknown"),
            "image_url": f"https://cdn2.thedogapi.com/images/{data.get('reference_image_id', '')}.jpg" if data.get("reference_image_id") else None
        }
        return info
    return {"error": "Breed not found"}

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        breed = request.form.get("breed")
        return redirect(url_for("breed_info", breed=breed))
    return render_template("index.html")

@app.route("/breed/<breed>")
def breed_info(breed):
    info = get_breed_info(breed)
    return render_template("result.html", info=info)

if __name__ == "__main__":
    app.run(debug=True)
'''

from flask import Flask, render_template, request, redirect, url_for
import base64
import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import img_to_array
import requests
import uuid  # For generating unique image filenames

app = Flask(__name__)
# Folder to save captured images
app.config['UPLOAD_FOLDER'] = 'Dog_breed_prediction/static/images/Captured_dog_imgs'
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Load the pretrained MobileNetV2 model (trained on ImageNet)
model = MobileNetV2(weights='imagenet')

def classify_breed(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image = img_to_array(image)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    preds = model.predict(image)
    # Decode predictions and take the top one
    breed = decode_predictions(preds, top=1)[0][0][1]
    return breed.replace('_', ' ')

def get_breed_info(breed):
    API_KEY = "live_okC9BHXOLCxf3vEaitiv4MQkwCdRHVPS4M5sjX9WPyxBZlbCa4i0ttDDEgf3QGOZ"  # Replace with your Dog API key
    url = f"https://api.thedogapi.com/v1/breeds/search?q={breed}"
    headers = {"x-api-key": API_KEY}
    response = requests.get(url, headers=headers)
    if response.status_code == 200 and response.json():
        data = response.json()[0]
        info = {
            "name": data.get("name", "Unknown"),
            "bred_for": data.get("bred_for", "Unknown"),
            "temperament": data.get("temperament", "Unknown"),
            "life_span": data.get("life_span", "Unknown"),
            "origin": data.get("origin", "Unknown"),
            "weight": (data.get("weight", {}).get("metric", "Unknown") + " kg") if data.get("weight") else "Unknown",
            "height": (data.get("height", {}).get("metric", "Unknown") + " cm") if data.get("height") else "Unknown",
            "breed_group": data.get("breed_group", "Unknown"),
            "image_url": f"https://cdn2.thedogapi.com/images/{data.get('reference_image_id', '')}.jpg" if data.get("reference_image_id") else None
        }
        return info
    return {"error": "Breed not found"}

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")
@app.route("/upload", methods=["POST"])
def upload():
    image_data = request.form.get("image_data")
    if image_data:
        # image_data is a base64 encoded string: "data:image/jpeg;base64,/9j/4AAQ..."
        header, encoded = image_data.split(",", 1)
        data = base64.b64decode(encoded)
        # Create a unique filename
        filename = f"{uuid.uuid4().hex}.jpg"
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        with open(image_path, "wb") as f:
            f.write(data)
        # Classify breed using the saved image
        breed = classify_breed(image_path)
        # Fetch detailed breed info from The Dog API
        info = get_breed_info(breed)
        return render_template("result.html", breed=breed, info=info, image_filename=filename)
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)

## http://127.0.0.1:5000/
