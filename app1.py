from flask import Flask, render_template, request, redirect, url_for
import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from PIL import Image
import base64
from io import BytesIO


app = Flask(__name__)


segmentation_model = load_model('C:\\Users\\kiran reddy\\Downloads\\major\\new_unet_brain_mri_seg (2).h5', custom_objects={
    'dice_coefficients_loss': lambda y_true, y_pred: -(2 * K.sum(y_true * y_pred) + 100) / (K.sum(y_true) + K.sum(y_pred) + 100),
    'iou': lambda y_true, y_pred: (K.sum(y_true * y_pred) + 100) / (K.sum(y_true + y_pred) - K.sum(y_true * y_pred) + 100),
})
detection_model = load_model('updatedBrainTumor10EpochsCategorical.h5')
classification_model = load_model('model.keras')

IM_HEIGHT, IM_WIDTH = 256, 256
DETECTION_INPUT_SIZE = 64
CLASS_NAMES = ['Glioma', 'Meningioma', 'No tumor', 'Pituitary'] 
DISPLAY_SIZE = (200, 200)  

# Helper function to convert image to base64
def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

# Segmentation function
def segment_image(image):
    img_resized = cv2.resize(image, (IM_WIDTH, IM_HEIGHT)) / 255.0
    img_input = np.expand_dims(img_resized, axis=0)
    pred_img = segmentation_model.predict(img_input)[0]
    pred_colored = cv2.applyColorMap((pred_img * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return pred_colored

# Detection function
def detect_tumor(image):
    img_resized = cv2.resize(image, (DETECTION_INPUT_SIZE, DETECTION_INPUT_SIZE)) / 255.0
    input_img = np.expand_dims(img_resized, axis=0)
    result = detection_model.predict(input_img)
    return np.argmax(result, axis=1)[0]

# Preprocess image for classification
def preprocess_image(image):
    size = (256, 256)
    image = image.resize(size)
    image = image.convert('L')  # Convert to grayscale
    image_array = np.array(image) / 255.0  # Normalize pixel values
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Classification function
def classify_image(image):
    image_array = preprocess_image(image)
    predictions = classification_model.predict(image_array)
    return np.argmax(predictions), predictions

@app.route('/')
def index():
    return render_template('index2.html')

@app.route('/segmentation', methods=['GET', 'POST'])
def segmentation():
    if request.method == 'POST':
        uploaded_file = request.files['image']
        if uploaded_file:
            try:
                # Load the uploaded image using PIL
                image = Image.open(uploaded_file)
                image_np = np.array(image)

                # Segment the image
                pred_colored = segment_image(image_np)
                pred_colored_resized = cv2.resize(pred_colored, (200, 200))

                # Convert images to base64
                base64_image = image_to_base64(Image.fromarray(image_np))
                base64_segmented_image = image_to_base64(Image.fromarray(pred_colored_resized))

                return render_template('segmentation.html', image=base64_image, segmented_image=base64_segmented_image)
            except Exception as e:
                return render_template('segmentation.html', error=f"Error: {e}")
    return render_template('segmentation.html')

@app.route('/detection', methods=['GET', 'POST'])
def detection():
    if request.method == 'POST':
        uploaded_file = request.files['image']
        if uploaded_file:
            try:
                # Load the uploaded image using PIL
                image = Image.open(uploaded_file)
                image_np = np.array(image)

                # Detect the tumor
                tumor_present = detect_tumor(image_np)
                tumor_status = "Tumor Detected" if tumor_present == 1 else "No Tumor Detected"

                # Convert image to base64
                base64_image = image_to_base64(Image.fromarray(image_np))

                return render_template('detection.html', image=base64_image, tumor_status=tumor_status)
            except Exception as e:
                return render_template('detection.html', error=f"Error: {e}")
    return render_template('detection.html')

@app.route('/classification', methods=['GET', 'POST'])
def classification():
    if request.method == 'POST':
        uploaded_file = request.files['image']
        if uploaded_file:
            try:
                # Load the uploaded image using PIL
                image = Image.open(uploaded_file)
                image_np = np.array(image)

                # Classify the tumor
                predicted_class_idx, _ = classify_image(image)
                predicted_class = CLASS_NAMES[predicted_class_idx]

                # Convert image to base64
                base64_image = image_to_base64(Image.fromarray(image_np))

                return render_template('classification.html', image=base64_image, prediction=predicted_class)
            except Exception as e:
                return render_template('classification.html', error=f"Error: {e}")
    return render_template('classification.html')

if __name__ == '__main__':
    app.run(debug=True)
