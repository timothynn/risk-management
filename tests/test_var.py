from risk_engine import RiskManager
import numpy as np

# Initialize with portfolio value
risk_mgr = RiskManager(portfolio_value=1_000_000)

# Calculate risk metrics from historical returns
returns = np.random.normal(0.001, 0.02, 1000)
metrics = risk_mgr.calculate_risk_metrics(returns)

print(f"95% VaR: {metrics.var_95*100:.2f}%")
print(f"99% VaR: {metrics.var_99*100:.2f}%")
print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
