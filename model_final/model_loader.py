import os
import numpy as np
import tensorflow as tf
from transformers import DistilBertTokenizerFast


class SentimentModel:
    def __init__(self):
        """
        Load the TFLite model and tokenizer.
        Set up input and output tensor details.
        """
        # Paths to the model and tokenizer
        base_path = os.path.dirname(__file__)
        self.model_path = os.path.join(base_path, "distilbert_model.tflite")
        tokenizer_path = os.path.join(base_path, "DISTILBERT_MODEL_TFLITE")

        # Load the tokenizer
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(
            tokenizer_path)

        # Load the TFLite model
        self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()

        # Retrieve input and output tensor details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def predict(self, texts):
        """
        Run inference on a list of input texts and return predictions.
        """
        # Tokenize input texts
        encodings = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="np"
        )

        # Convert inputs to int32 (required by TFLite)
        input_ids = encodings["input_ids"].astype("int32")
        attention_mask = encodings["attention_mask"].astype("int32")

        # Assign inputs to interpreter tensors
        for input_detail in self.input_details:
            input_name = input_detail["name"]
            if "input_ids" in input_name:
                self.interpreter.set_tensor(input_detail["index"], input_ids)
            elif "attention_mask" in input_name:
                self.interpreter.set_tensor(
                    input_detail["index"], attention_mask)

        # Run inference
        self.interpreter.invoke()

        # Get prediction from output tensor
        output = self.interpreter.get_tensor(self.output_details[0]["index"])
        predictions = np.argmax(output, axis=1)

        return predictions.tolist()
