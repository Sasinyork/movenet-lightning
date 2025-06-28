import tensorflow as tf
import tensorflow_hub as hub
import os
import requests

def download_tflite_model(url, filename="model.tflite"):
    """Downloads the TFLite model file from TensorFlow Hub."""
    if not os.path.exists(filename):
        response = requests.get(url)
        with open(filename, "wb") as f:
            f.write(response.content)

def load_movenet_model(model_name="movenet_lightning"):
    """Loads MoveNet model and returns (movenet_fn, input_size)."""
    if "tflite" in model_name:
        if "lightning_f16" in model_name:
            url = "https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/float16/4?lite-format=tflite"
            input_size = 192
        elif "thunder_f16" in model_name:
            url = "https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/tflite/float16/4?lite-format=tflite"
            input_size = 256
        elif "lightning_int8" in model_name:
            url = "https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/int8/4?lite-format=tflite"
            input_size = 192
        elif "thunder_int8" in model_name:
            url = "https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/tflite/int8/4?lite-format=tflite"
            input_size = 256
        else:
            raise ValueError("Unsupported model name: %s" % model_name)

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

    else:
        if "lightning" in model_name:
            module = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
            input_size = 192
        elif "thunder" in model_name:
            module = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
            input_size = 256
        else:
            raise ValueError("Unsupported model name: %s" % model_name)

        model = module.signatures['serving_default']

        def movenet(input_image):
            input_image = tf.cast(input_image, dtype=tf.int32)
            outputs = model(input_image)
            return outputs['output_0'].numpy()

        return movenet, input_size
