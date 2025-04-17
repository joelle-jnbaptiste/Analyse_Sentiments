from model_final.model_loader import SentimentModel


def test_prediction_positive():
    """
    Test that a clearly positive sentence is classified as positive (1).
    """
    model = SentimentModel()
    input_text = ["This movie was fantastic and inspiring"]
    prediction = model.predict(input_text)

    assert isinstance(prediction, list)
    assert isinstance(prediction[0], int)
    assert prediction[0] == 1


def test_prediction_negative():
    """
    Test that a clearly negative sentence is classified as negative (0).
    """
    model = SentimentModel()
    input_text = ["I hated this movie, it was terrible"]
    prediction = model.predict(input_text)

    assert isinstance(prediction, list)
    assert isinstance(prediction[0], int)
    assert prediction[0] == 0


def test_batch_prediction_accuracy():
    """
    Test batch prediction accuracy with known positive and negative samples.
    Ensures the model achieves at least 80% accuracy on this sample.
    """
    model = SentimentModel()

    positive_sentences = [
        "What a great experience!",
        "I absolutely loved it.",
        "This was amazing!",
        "Incredible performance by the cast.",
        "A wonderful and touching story."
    ]

    negative_sentences = [
        "It was a complete waste of time.",
        "Terrible plot and poor acting.",
        "I didn't like it at all.",
        "So boring, I almost fell asleep.",
        "Absolutely awful experience."
    ]

    all_sentences = positive_sentences + negative_sentences
    expected = [1] * len(positive_sentences) + [0] * len(negative_sentences)

    predictions = [model.predict([sentence])[0] for sentence in all_sentences]

    correct = sum(pred == exp for pred, exp in zip(predictions, expected))
    accuracy = correct / len(expected)

    assert accuracy >= 0.8, f"Accuracy too low: {accuracy * 100:.1f}%"
