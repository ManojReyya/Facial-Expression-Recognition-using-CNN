import os
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Load the pre-trained model
model = load_model("Expression.h5")

# Define the target size for resizing images
target_size = (64, 64)

@app.route('/')
def index():
    return render_template("index.html", emotion="")

@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        # Get the file from the POST request
        f = request.files['image']
        
        # Save the file to the uploads folder
        uploads_dir = os.path.join(app.root_path, 'uploads')
        if not os.path.exists(uploads_dir):
            os.makedirs(uploads_dir)
        
        filepath = os.path.join(uploads_dir, f.filename)
        f.save(filepath)
        
        # Preprocess the image
        img = image.load_img(filepath, target_size=target_size)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x / 255.0  # Normalize pixel values
        
        # Print shape of x for debugging
        print("Shape of input image array:", x.shape)
        
        # Make prediction
        try:
            pred = model.predict(x)
            pred_class = np.argmax(pred, axis=1)
            index = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
            emotion = index[pred_class[0]]
            return render_template("index.html", emotion=emotion)
        except Exception as e:
            print("Error during prediction:", str(e))
            return render_template("index.html", emotion="Error during prediction")

if __name__ == '__main__':
    app.run(debug=True)
