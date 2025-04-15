from transformers import DistilBertTokenizerFast
import tensorflow as tf
import numpy as np
import os

class SentimentModel:
    def __init__(self):
        # Chemins vers le modèle TFLite et le tokenizer
        base_path = os.path.dirname(__file__)
        self.model_path = os.path.join(base_path, "distilbert_model.tflite")
        tokenizer_path = os.path.join(base_path, "DISTILBERT_MODEL_TFLITE")

        # Chargement du tokenizer
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(tokenizer_path)

        # Chargement du modèle TFLite
        self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()

        # Détails des entrées et sorties
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def predict(self, texts):
        # Prétraitement
        encodings = self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors="np"
        )

        # Cast des entrées en int32 (TFLite est strict là-dessus)
        input_ids = encodings["input_ids"].astype("int32")
        attention_mask = encodings["attention_mask"].astype("int32")

        # Assignation des tensors selon le nom des entrées
        for input_detail in self.input_details:
            input_name = input_detail["name"]
            if "input_ids" in input_name:
                self.interpreter.set_tensor(input_detail["index"], input_ids)
            elif "attention_mask" in input_name:
                self.interpreter.set_tensor(input_detail["index"], attention_mask)

        # Inférence
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        predictions = np.argmax(output, axis=1)
        return predictions.tolist()

