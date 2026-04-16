import numpy as np

class Metrics:
    @staticmethod
    def calculate_returns(values):
        return np.diff(values) / values[:-1]

    @staticmethod
    def sharpe_ratio(returns):
        return np.mean(returns) / (np.std(returns) + 1e-9)

    @staticmethod
    def max_drawdown(values):
        peak = values[0]
        max_dd = 0

        for v in values:
            if v > peak:
                peak = v
            dd = (peak - v) / peak
            if dd > max_dd:
                max_dd = dd

        return max_dd