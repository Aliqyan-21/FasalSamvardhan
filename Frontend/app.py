import os
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['STATIC_FOLDER'] = 'static/images'

# Model Loading
MODEL = tf.keras.models.load_model('../saved_models/3')
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

# Define the allowed file extensions
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'webp'])

# Function to check if a file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':

        # Check if a file was uploaded
        if 'file' not in request.files:
            return render_template('upload.html', error='No file selected')
        
        file = request.files['file']
        
        # Check if the file has an allowed extension
        if not allowed_file(file.filename):
            return render_template('upload.html', error='File type not allowed')
        
        # Save the file to the uploads directory
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Save the file to the static directory
        img = Image.open(file_path)
        img_path = os.path.join(app.config['STATIC_FOLDER'], filename)
        img.save(img_path)

        return redirect(url_for('predict', file_path=img_path))
    
    return render_template('upload.html')

@app.route('/predict')
def predict():
    file_path = request.args.get('file_path')

    img = Image.open(file_path)

    img_batch = np.expand_dims(img, 0)

    try:
        img_batch = tf.image.resize(img_batch, (256, 256))
    except:
        error_message = "Image could not be processed. Please upload a valid image."
        return render_template("result.html", error_message=error_message)

    try:
        predictions = MODEL.predict(img_batch)
    except:
        error_message = "Image could not be processed. Please upload a valid image."
        return render_template("result.html", error_message=error_message)

    index = np.argmax(predictions[0])
    predicted_class = CLASS_NAMES[index]
    confidence = 100*np.max(predictions[0])
    confidence = round(confidence, 2)

    return render_template("result.html", predicted_class = predicted_class, confidence=confidence, img_url = file_path)