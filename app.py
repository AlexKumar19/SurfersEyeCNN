from flask import Flask, render_template, request, redirect, url_for
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
import cv2
import os
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load your model
model = Sequential()
model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256,256,3)))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(16, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.load_weights('model.h5')

@app.route('/')
@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']

    # Ensure the static directory exists
    if not os.path.exists('static'):
        os.makedirs('static')

    filepath = os.path.join('static', file.filename)
    file.save(filepath)
    
    image = Image.open(filepath).convert('RGB')
    image = np.array(image)
    image = cv2.resize(image, (256,256))
    image = np.expand_dims(image / 255.0, axis=0)  # Normalize and add batch dimension
    
    yhat = model.predict(image)
    # result = ""
    # if yhat[0][0] > 0.5: 
    #     result = 'Predicted class is normal'
    # else:
    #     result = 'Predicted class surfer'
    
    result = 'Predicted class is normal' if yhat[0][0] > 0.5 else 'Predicted class surfer'
    return jsonify({'result': result, 'image_url': url_for('static', filename=file.filename)})


@app.route('/result/<filename>/<result>')
def result(filename, result):
    return render_template('result.html', filename=filename, result=result)

if __name__ == '__main__':
    app.run(debug=True)
