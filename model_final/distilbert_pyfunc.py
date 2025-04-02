import mlflow.pyfunc
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd

class DistilBertPyFunc(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        if "<OOV>" not in self.tokenizer.get_vocab():
            self.tokenizer.add_tokens(["<OOV>"])

        self.model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
        self.model.resize_token_embeddings(len(self.tokenizer))

        model_path = context.artifacts["state_dict"]
        state_dict = torch.load(model_path, map_location=torch.device("cpu"), weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict(self, context, model_input: pd.DataFrame):
        texts = model_input["text"].tolist()
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)

        return preds.numpy()
