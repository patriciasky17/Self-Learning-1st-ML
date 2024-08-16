from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np

from tensorflow.keras.layers import DepthwiseConv2D
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_custom_objects

# Custom DepthwiseConv2D to handle the 'groups' argument
class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        if 'groups' in kwargs:
            del kwargs['groups']
        super().__init__(*args, **kwargs)

# Register the custom layer
get_custom_objects().update({'DepthwiseConv2D': CustomDepthwiseConv2D})

app = Flask(__name__)

# Load your Keras model
model = load_model('model/keras_model.h5', compile=False)

# Load the labels
with open('model/labels.txt', 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

def prepare_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Read the image file
        file = request.files["file"]
        if not file:
            return "No file uploaded!", 400

        # Prepare the image for classification
        image = Image.open(file)
        processed_image = prepare_image(image, target_size=(224, 224))

        # Make prediction
        predictions = model.predict(processed_image).flatten()
        top_prediction = np.argmax(predictions)
        label = class_names[top_prediction]
        confidence = predictions[top_prediction]

        return jsonify({
            "label": label,
            "confidence": float(confidence)
        })

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)