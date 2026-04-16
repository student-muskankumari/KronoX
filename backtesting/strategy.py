class SimpleStrategy:
    def __init__(self, threshold=0.01):
        self.threshold = threshold

    def decide(self, current_price, predicted_price):
        if predicted_price > current_price * (1 + self.threshold):
            return "BUY"
        elif predicted_price < current_price * (1 - self.threshold):
            return "SELL"
        return "HOLD"