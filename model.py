import tflite_runtime.interpreter as tflite
import numpy as np
import cv2
import os
import time


class TFLiteModel:

    def __init__(self, model_path, labels_path):
        self.model_path = os.path.expanduser(model_path)
        self.labels_path = os.path.expanduser(labels_path)
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.labels = []

        self._load_model()
        self._load_labels()

    def _load_model(self):
        """Loads the TFLite model. Will raise an error if loading fails."""
        self.interpreter = tflite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        print("TFLite model loaded successfully.")

    def _load_labels(self):
        """Loads class labels from a file. Will raise an error if loading fails."""
        with open(self.labels_path, 'r') as f:
            self.labels = f.read().splitlines()
        print(f"Labels loaded: {self.labels}")

    def preprocess_image(self, image):
        """
        Preprocesses the image to match the model's expected input.
        Adjust this function based on your model's specific requirements!
        """
        if self.input_details is None:
            raise RuntimeError("Model not loaded. Cannot preprocess image.")

        input_shape = self.input_details[0]['shape'] # e.g., (1, 224, 224, 3)
        input_height = input_shape[1]
        input_width = input_shape[2]
        input_dtype = self.input_details[0]['dtype']

        # Resize the image
        resized_image = cv2.resize(image, (input_width, input_height))

        # Add batch dimension
        input_data = np.expand_dims(resized_image, axis=0)

        # Normalize based on expected dtype
        if input_dtype == np.float32:
            input_data = input_data.astype(np.float32) / 255.0
        elif input_dtype == np.uint8:
            input_data = input_data.astype(np.uint8)
        else:
            raise ValueError(f"Unsupported input data type: {input_dtype}")

        return input_data

    def predict(self, image):
        """
        Performs inference on a single image.
        Returns the predicted label, confidence, and inference time.
        Raises RuntimeError if interpreter is not loaded.
        """
        if self.interpreter is None:
            raise RuntimeError("Model interpreter is not loaded. Cannot perform prediction.")

        processed_input = self.preprocess_image(image)

        self.interpreter.set_tensor(self.input_details[0]['index'], processed_input)

        inference_start_time = time.time()
        self.interpreter.invoke()
        inference_end_time = time.time()
        inference_time_ms = (inference_end_time - inference_start_time) * 1000

        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        prediction = np.squeeze(output_data)

        # If your model's output is uint8 (from full integer quantization), dequantize
        if self.output_details[0]['dtype'] == np.uint8:
            scale, zero_point = self.output_details[0]['quantization']
            if scale != 0:
                prediction = (prediction.astype(np.float32) - zero_point) * scale

        predicted_label = "Unknown"
        confidence = 0.0

        if self.labels:
            predicted_class_id = np.argmax(prediction)
            predicted_label = self.labels[predicted_class_id]
            confidence = prediction[predicted_class_id]
            # Consider applying softmax here if your model outputs logits:
            # exp_pred = np.exp(prediction - np.max(prediction))
            # probabilities = exp_pred / np.sum(exp_pred)
            # confidence = probabilities[predicted_class_id]
        else:
            predicted_label = f"Raw Output: {prediction[:5]}..."

        return predicted_label, confidence, inference_time_ms
