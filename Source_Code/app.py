from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
import os
from werkzeug.utils import secure_filename

# ========================== Flask App Setup ==========================
app = Flask(__name__)

# Folder to store uploaded images
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ========================== Load Model ==========================
MODEL_PATH = 'my_model.h5'   # Your fine-tuned model file
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully!")
print("Input shape:", model.input_shape)
print("Output shape:", model.output_shape)

# ========================== Class Labels ==========================
# Must match the order used during training
CLASS_NAMES = [
    'Melanocytic Nevi',                # nv
    'Melanoma',                        # mel
    'Benign keratosis-like lesions',   # bkl
    'Basal cell carcinoma',            # bcc
    'Actinic keratoses',               # akiec
    'Vascular lesions',                # vasc
    'Dermatofibroma'                   # df
]

# ========================== Routes ==========================
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', prediction_text="No file uploaded!")

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', prediction_text="No file selected!")

    # Save file to static/uploads
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # ================== Preprocess Image ==================
    img = image.load_img(filepath, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)  # IMPORTANT FIX for ResNet50

    # ================== Make Prediction ==================
    preds = model.predict(img_array)
    predicted_index = np.argmax(preds)
    confidence = float(np.max(preds))
    predicted_class = CLASS_NAMES[predicted_index]

    # ================== Debug Info ==================
    print("\n----- Prediction Log -----")
    print("Raw Predictions:", preds)
    print("Predicted Index:", predicted_index)
    print("Predicted Class:", predicted_class)
    print("Confidence:", confidence)
    print("---------------------------\n")

    # Optional: handle low confidence predictions
    if confidence < 0.5:
        predicted_class = "Uncertain / Low confidence"

    # Render result page
    return render_template(
        'index.html',
        prediction_text=f"Prediction: {predicted_class} (Confidence: {confidence:.2f})",
        img_path=filepath
    )


@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


# ========================== Run App ==========================
if __name__ == '__main__':
    app.run(debug=True)
