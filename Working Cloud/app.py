from flask import Flask, render_template, request
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import Image
import os
import numpy as np
import cv2

app = Flask(__name__)

# Create 'uploads' directory if it doesn't exist
upload_folder = 'static/uploads'
os.makedirs(upload_folder, exist_ok=True)

# Load the pre-trained model
model = load_model('DeepFakeMajor_88%.h5')

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (64, 64))  # Resize the image to match the model input shape
    img_array = np.array(img)
    img_array = img_array / 255.0  # Normalize the pixel values to the range [0, 1]
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to make predictions
def make_prediction(image_path):
    img_array = preprocess_image(image_path)
    predictions = model.predict(img_array)
    return predictions

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the uploaded file
        file = request.files['file']
        
        if file:
            # Save the uploaded file in the 'uploads' directory
            file_path = os.path.join(upload_folder, file.filename)
            file.save(file_path)

            # Make predictions
            predictions = make_prediction(file_path)

            # Decode predictions (modify based on your model output)
            class_labels = ['The image is Real', 'The image is DeepFake']  # Replace with your class labels
            predicted_class = class_labels[np.argmax(predictions)]

            return render_template('index.html', image_path=file_path, predicted_class=predicted_class)

    return render_template('index.html', image_path=None, predicted_class=None)

if __name__ == '__main__':
    app.run(debug=True)
