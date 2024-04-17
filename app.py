from flask import Flask, request, jsonify
import numpy as np
print(np.__version__)
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import io


# Load the model
modelVgg16 = load_model('model_vgg.h5')
modelResNet50 = load_model('model_resnet50.h5')
modelInceptionV3 = load_model('model_inception_v3.h5')

# Create a Flask app
app = Flask(__name__)

@app.route('/predict_vgg', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'no file'}), 400

    file = request.files['file'].read()
    img = load_img(io.BytesIO(file), target_size=(448, 448))  # Update target size to (448, 448)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img/255

    prediction = modelVgg16.predict(img)
    print(prediction)
    prediction = np.where(prediction > 0.6, 1, 0)
    data = {'class': int(prediction[0][0])}

    return jsonify(data)

@app.route('/predict_resnet', methods=['POST'])
def predict_resnet():
    if 'file' not in request.files:
        return jsonify({'error': 'no file'}), 400

    file = request.files['file'].read()
    img = load_img(io.BytesIO(file), target_size=(448, 448))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img/255

    prediction = modelResNet50.predict(img)
    prediction = np.where(prediction > 0.5, 1, 0)
    data = {'class': int(prediction[0][0])}

    return jsonify(data)

@app.route('/predict_inception', methods=['POST'])
def predict_inception():
    if 'file' not in request.files:
        return jsonify({'error': 'no file'}), 400

    file = request.files['file'].read()
    img = load_img(io.BytesIO(file), target_size=(448,448))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img/255

    prediction = modelInceptionV3.predict(img)
    prediction = np.where(prediction > 0.5, 1, 0)
    data = {'class': int(prediction[0][0])}

    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True, port=7860, host='0.0.0.0')