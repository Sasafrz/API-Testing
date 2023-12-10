from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

app = Flask(__name__)
model = load_model('model_ml.h5')

# Data dummy sebagai contoh
semaphore_data = [
    {'id': 1, 'label': 'A'},
    {'id': 2, 'label': 'B'},
    # ... tambahkan data semaphore lainnya ...
]


@app.route('/api/semaphore', methods=['GET'])
def get_all_semaphores():
    response_data = [{'id': item['id'], 'label': item['label']}
                     for item in semaphore_data]
    return jsonify({'semaphores': response_data})


@app.route('/api/semaphore/<int:id>', methods=['GET'])
def get_semaphore_detail(id):
    semaphore = next(
        (item for item in semaphore_data if item['id'] == id), None)
    if semaphore:
        return jsonify({'semaphore': semaphore})
    else:
        return jsonify({'message': 'Semaphore not found'}), 404


def recognize_letter(img):
    # Resize gambar ke ukuran yang diharapkan oleh model
    img = Image.open(img).convert("L")
    img = np.array(img.resize((28, 28)))

    # Normalisasi
    img = img / 255.0

    # Prediksi huruf dari gambar
    prediction = model.predict(np.array([img]))
    predicted_index = np.argmax(prediction[0])
    predicted_letter = chr(ord('A') + predicted_index)

    # Jika huruf hasil prediksi berada di luar A-Z, atur menjadi karakter kosong
    if not 'A' <= predicted_letter <= 'Z':
        predicted_letter = ''

    return predicted_letter


@app.route('/api/semaphore/recognize', methods=['POST'])
def recognize_semaphore():
    try:
        # Ambil gambar dari input pengguna
        image = request.files['image']

        # Prediksi huruf dari gambar
        predicted_letter = recognize_letter(image)

        return jsonify({'predicted_letter': predicted_letter})
    except KeyError:
        return jsonify({'error': 'No image provided'}), 400


if __name__ == '__main__':
    app.run(debug=True)
