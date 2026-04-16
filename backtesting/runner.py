import pandas as pd
from backtesting.engine import BacktestEngine
from backtesting.strategy import SimpleStrategy
from backtesting.metrics import Metrics

def run_backtest(csv_path):
    df = pd.read_csv(csv_path)

    engine = BacktestEngine()
    strategy = SimpleStrategy()

    portfolio_values = []

    for i, row in df.iterrows():
        price = row["actual_price"]
        predicted = row["predicted_price"]
        date = row.get("date", i)

        action = strategy.decide(price, predicted)
        engine.step(date, price, action)

        portfolio_values.append(engine.get_portfolio_value(price))

    returns = Metrics.calculate_returns(portfolio_values)

    return {
        "final_capital": portfolio_values[-1],
        "return_pct": ((portfolio_values[-1] / engine.initial_capital) - 1) * 100,
        "sharpe": Metrics.sharpe_ratio(returns),
        "drawdown": Metrics.max_drawdown(portfolio_values),
        "trades": len(engine.trade_log)
    }