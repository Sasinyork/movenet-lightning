import tensorflow as tf
import os
import requests

def download_tflite_model(url, filename="model.tflite"):
    """Downloads the TFLite model file from TensorFlow Hub."""
    if not os.path.exists(filename):
        response = requests.get(url)
        with open(filename, "wb") as f:
            f.write(response.content)

def load_movenet_model():
    """Loads MoveNet Lightning TFLite model optimized for mobile/Android."""
    # Lightning TFLite model (optimized for mobile)
    url = "https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/float16/4?lite-format=tflite"
    input_size = 192

    download_tflite_model(url)

    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()

    def movenet(input_image):
        input_image = tf.cast(input_image, dtype=tf.uint8)
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.set_tensor(input_details[0]['index'], input_image.numpy())
        interpreter.invoke()
        keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
        return keypoints_with_scores

    return movenet, input_size
