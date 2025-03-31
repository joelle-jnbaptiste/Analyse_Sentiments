# model_final/model_loader.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class SentimentModel:
    def __init__(self, model_path: str = "model_final/DISTILBERT_MODEL_FULLY_TRAINED.pt"):
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        if "<OOV>" not in self.tokenizer.get_vocab():
            self.tokenizer.add_tokens(["<OOV>"])
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=2
        )
        self.model.resize_token_embeddings(len(self.tokenizer))
        state_dict = torch.load(model_path, map_location=torch.device("cpu"))
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict(self, texts):
        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
        return preds.numpy()
