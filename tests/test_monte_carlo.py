from risk_engine import RiskManager
import numpy as np

# Portfolio setup
initial_value = 1_000_000
weights = np.array([0.4, 0.3, 0.2, 0.1])  # Asset allocation

# Market parameters
expected_returns = np.array([0.0008, 0.0006, 0.0004, 0.0005])
covariance = np.eye(4) * 0.0004  # Simplified covariance

# Run simulation
risk_mgr = RiskManager(initial_value)
simulations, stats = risk_mgr.run_monte_carlo_analysis(
    weights, expected_returns, covariance, time_horizon=252
)

print(f"Mean Final Value: ${stats['mean_final_value']:,.2f}")
print(f"95% VaR: {stats['var_95']*100:.2f}%")
