from flask import Flask, render_template, request
from flask_bootstrap import Bootstrap
from PIL import Image
import tensorflow as tf
import numpy as np
import os
from datetime import datetime

app = Flask(__name__)
Bootstrap(app)

# Folder for uploaded images
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

try:
    model = tf.keras.models.load_model('Model/SimpleCNN2_fer_model.h5')
    emotion_labels = ['Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Neutral']
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

def preprocess_image(image):
    img = image.convert('L')
    img = img.resize((48, 48), Image.Resampling.LANCZOS)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    img_array = np.expand_dims(img_array, axis=-1)
    return img_array

def predict_emotion(image):
    processed_img = preprocess_image(image)
    prediction = model.predict(processed_img)
    emotion_idx = np.argmax(prediction)
    return emotion_labels[emotion_idx]

@app.route('/', methods=['GET', 'POST'])
def index():
    emotion = None
    image_path = None
    if request.method == 'POST':
        if 'file' in request.files:
            file = request.files['file']
            if file and file.filename != '':
                timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
                filename = f"uploaded_{timestamp}.png"
                file_path = os.path.join(UPLOAD_FOLDER, filename)
                file.save(file_path)
                image = Image.open(file_path)
                emotion = predict_emotion(image)
                image_path = filename
    return render_template('Index.html', emotion=emotion, image_path=image_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
