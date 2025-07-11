import tensorflow as tf
import os
import requests

def download_tflite_model(url, filename="model.tflite"):
    """Downloads the TFLite model file from TensorFlow Hub."""
    if not os.path.exists(filename):
        response = requests.get(url)
        with open(filename, "wb") as f:
            f.write(response.content)

def load_movenet_model(custom_model_path=None):
    """Loads MoveNet Lightning TFLite model optimized for mobile/Android."""
    
    if custom_model_path and os.path.exists(custom_model_path):
        # Use custom trained model
        print(f"Loading custom trained model: {custom_model_path}")
        model_path = custom_model_path
    else:
        # Use pre-trained model
        print("Loading pre-trained MoveNet Lightning model...")
        url = "https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/float16/4?lite-format=tflite"
        download_tflite_model(url)
        model_path = "model.tflite"
    
    input_size = 192  # Default for Lightning model

    interpreter = tf.lite.Interpreter(model_path=model_path)
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
