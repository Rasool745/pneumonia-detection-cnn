import os
import numpy as np
from PIL import Image
import cv2
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
from tensorflow.keras.applications.vgg19 import VGG19
from huggingface_hub import snapshot_download  # <--- Added this

# --- NEW: Hugging Face Connection ---
repo_id = "Rasool786/pneumonia-weights"
local_weights_dir = "model_weights"

# Check if weights exist locally; if not, download from Hugging Face
if not os.path.exists(local_weights_dir):
    print("Fetching model weights from Hugging Face... this may take a few minutes.")
    snapshot_download(repo_id=repo_id, local_dir=local_weights_dir, repo_type="model")
# -------------------------------------

# Load VGG19 base model
base_model = VGG19(include_top=False, input_shape=(128, 128, 3))

x = base_model.output
flat = Flatten()(x)
class_1 = Dense(4608, activation='relu')(flat)
drop_out = Dropout(0.2)(class_1)
class_2 = Dense(1152, activation='relu')(drop_out)
output = Dense(2, activation='softmax')(class_2)

model_03 = Model(inputs=base_model.inputs, outputs=output)

# Load trained weights from the downloaded folder
# Note: snapshot_download might place them in a subfolder depending on your upload structure
weights_file = os.path.join(local_weights_dir, "vgg_unfrozen.h5")
model_03.load_weights(weights_file)

# Create Flask app
app = Flask(__name__)

# Convert class number to label
def get_className(classNo):
    if classNo == 0:
        return "NORMAL"
    elif classNo == 1:
        return "PNEUMONIA DETECTED"

# Prediction function
def getResult(img_path):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = image.resize((128, 128))

    image = np.array(image)
    image = image.astype('float32') / 255.0

    input_img = np.expand_dims(image, axis=0)

    preds = model_03.predict(input_img)
    print(f"DEBUG - Raw Scores: {preds}")

    result_index = np.argmax(preds, axis=1)
    return result_index[0]

# Home page
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# Upload and prediction
@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        upload_path = os.path.join(basepath, 'uploads')

        if not os.path.exists(upload_path):
            os.makedirs(upload_path)

        file_path = os.path.join(upload_path, secure_filename(f.filename))
        f.save(file_path)

        value = getResult(file_path)
        result_text = get_className(value)

        return result_text

    return None

if __name__ == '__main__':
    # host='0.0.0.0' allows the cloud server to access the app
    app.run(host='0.0.0.0', port=7860)