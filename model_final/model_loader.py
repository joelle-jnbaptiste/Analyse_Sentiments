from transformers import TFDistilBertForSequenceClassification, DistilBertTokenizerFast
import tensorflow as tf
import os

class SentimentModel:
    def __init__(self):
        # Dossier contenant le modèle TensorFlow et le tokenizer
        model_path = os.path.join(os.path.dirname(__file__), "DISTILBERT_MODEL_TF_FULL")
        
        # Chargement du tokenizer et du modèle
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
        self.model = TFDistilBertForSequenceClassification.from_pretrained(model_path)
    
    def predict(self, texts):
        # Tokenisation
        inputs = self.tokenizer(texts, return_tensors="tf", padding=True, truncation=True)

        # Prédiction sans gradients
        outputs = self.model(inputs)
        logits = outputs.logits
        predictions = tf.argmax(logits, axis=1)

        return predictions.numpy().tolist()
