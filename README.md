### Documentation for Self-Learning Projects (Braincore)

---

#### **Project 1: WebPyTorch (YOLOv8 for Rock-Paper-Scissors Object Detection)**

**Overview:**
This project is designed to identify rock, paper, or scissors gestures using object detection. It leverages the YOLOv8 model from Ultralytics, implemented in PyTorch.

**Source Dataset:**
- The dataset used for training the YOLOv8 model was sourced from Roboflow: [Rock Paper Scissors SXSW Dataset](https://universe.roboflow.com/roboflow-58fyf/rock-paper-scissors-sxsw/dataset/14).

**Folder Structure and Explanation:**
- **models/**: Contains the trained YOLOv8 model used for object detection.
- **static/**: Houses CSS and JavaScript files required for the frontend.
- **templates/**: Contains the HTML files used to render the web interface.
- **TestResult/**: Stores test results of the model using images not seen during training. Each entry includes one image and its corresponding detection result.
- **training_logs/**: This folder contains the training weights and logs generated during the model training process.

**How to Run:**
1. Navigate to the `WebPytorch` directory.
2. Run the command: 
   ```bash
   python application.py
   ```
3. Access the application via the localhost URL provided after running the script.

**Purpose:**
The primary goal is to detect and classify gestures (rock, paper, scissors) using a pre-trained YOLOv8 model. The interface allows users to upload an image and view the detection results in real-time.

---

#### **Project 2: WebTensorflow (Teachable Machine for Flower Classification)**

**Overview:**
This project is focused on image classification, identifying five different types of flowers: Tulip, Daisy, Sunflower, Dandelion, and Rose. The model is trained using Google's Teachable Machine and exported to TensorFlow.js.

**Source Dataset:**
- The dataset used for training the model in Teachable Machine was sourced from Kaggle: [Flowers Dataset](https://www.kaggle.com/datasets/imsparsh/flowers-dataset).

**Folder Structure and Explanation:**
- **model/**: Contains the TensorFlow.js model files exported from Teachable Machine.
- **proof/**: Includes screenshots taken from Teachable Machine showing the model's training process and results.

**How to Run:**
1. Open the `WebTensorflow` folder.
2. Start a live server (e.g., through VS Code or another IDE that supports live server functionality).
3. Open `index.html` in a web browser to interact with the application.

**Purpose:**
This project aims to classify images of flowers into five categories using a pre-trained model from Teachable Machine. The user interface allows for uploading an image and getting a classification result instantly.
