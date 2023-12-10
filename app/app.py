from flask import Flask, jsonify, request
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
import numpy as np

app = Flask(__name__)

# Load your trained model
# (Make sure to replace 'your_model.h5' with the actual name of your saved model file)
model_path = '/content/drive/your_model.h5'
model = tf.keras.models.load_model(model_path)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image file from the request
        file = request.files['image']
        
        # Save the image to a temporary file
        temp_path = '/tmp/temp_image.jpg'
        file.save(temp_path)

        # Preprocess the image
        img = image.load_img(temp_path, target_size=(150, 150))
        img_array = image.img_to_array(img)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        # Make predictions
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions)
        predicted_letter = chr(predicted_class + ord("A"))

        # Return the prediction as JSON
        response = {'predicted_class': predicted_letter}
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
