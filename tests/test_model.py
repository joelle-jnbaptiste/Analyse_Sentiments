from Analyse_Sentiments.model_final.model_loader import SentimentModel


def test_model_prediction():
    model = SentimentModel()
    sample_input = ["It was amazing"]
    prediction = model.predict(sample_input)
    
    assert isinstance(prediction, list)
    assert isinstance(prediction[0], int)
    assert prediction[0] in [0, 1]
