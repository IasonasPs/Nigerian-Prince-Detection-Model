

class PredictionResult:
    def __init__(self, prediction, probability):
        self.prediction = prediction
        self.probability = probability

    def __str__(self):
        return f"Prediction: {self.prediction}, Probability: {self.probability:.2f}"

    def __repr__(self):
        return f"PredictionResult(prediction={self.prediction}, probability={self.probability})"