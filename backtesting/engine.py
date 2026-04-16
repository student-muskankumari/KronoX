class BacktestEngine:
    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.position = 0
        self.entry_price = 0
        self.trade_log = []

    def step(self, date, price, action):
        if action == "BUY" and self.capital > 0:
            self.position = self.capital // price
            self.entry_price = price
            self.capital -= self.position * price

            self.trade_log.append({
                "date": date,
                "action": "BUY",
                "price": price,
                "shares": self.position
            })

        elif action == "SELL" and self.position > 0:
            self.capital += self.position * price

            self.trade_log.append({
                "date": date,
                "action": "SELL",
                "price": price,
                "shares": self.position
            })

            self.position = 0
            self.entry_price = 0

    def get_portfolio_value(self, current_price):
        return self.capital + self.position * current_price