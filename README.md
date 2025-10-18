# Risk Management System

A comprehensive, high-performance risk management system built with Python and Nix flakes. Features Value at Risk (VaR) calculation, Conditional VaR (CVaR), Monte Carlo simulation, and stress testing capabilities optimized for financial analysis.

## Features

### 🎯 Core Capabilities
- **Value at Risk (VaR)**: Historical, Parametric, and Monte Carlo methods
- **Conditional VaR (CVaR)**: Expected Shortfall calculations
- **Monte Carlo Simulation**: High-performance portfolio simulations with JIT compilation
- **Stress Testing**: Market crash and rate shock scenarios
- **Risk Metrics**: Sharpe ratio, volatility, maximum drawdown

### ⚡ Performance
- Numba JIT compilation for critical calculations
- Optimized for high-frequency risk calculations
- Efficient matrix operations with NumPy
- Parallel simulation capabilities

## Project Structure

```
risk-management-system/
├── flake.nix              # Nix flake configuration
├── flake.lock             # Locked dependencies
├── README.md              # This file
├── src/
│   ├── main.py            # Main application
│   ├── risk_engine.py     # Core risk calculation engine
│   └── __init__.py
└── tests/
    ├── test_var.py
    ├── test_monte_carlo.py
    └── test_stress_testing.py
```

## Getting Started

### Prerequisites
- Nix package manager with flakes enabled

### Installation

1. **Clone the repository**:
```bash
git clone <your-repo-url>
cd risk-management-system
```

2. **Enable Nix flakes** (if not already enabled):
```bash
# Add to ~/.config/nix/nix.conf or /etc/nix/nix.conf
experimental-features = nix-command flakes
```

3. **Enter development environment**:
```bash
nix develop
```

### Running the System

**Inside the development shell**:
```bash
python src/main.py
```

**Or using the Nix app**:
```bash
nix run
```

**Build the package**:
```bash
nix build
./result/bin/risk-system
```

## Usage Examples

### Basic VaR Calculation
```python
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
```

### Monte Carlo Simulation
```python
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
```

### Stress Testing
```python
from risk_engine import RiskManager

# Initialize portfolio
risk_mgr = RiskManager(1_000_000)
risk_mgr.add_position('equities', 400_000)
risk_mgr.add_position('bonds', 300_000)
risk_mgr.add_position('real_estate', 200_000)
risk_mgr.add_position('commodities', 100_000)

# Run stress tests
stress_results = risk_mgr.run_stress_tests()
for scenario, pnl_pct in stress_results.items():
    print(f"{scenario}: {pnl_pct:+.2f}%")
```

## Risk Metrics Explained

### Value at Risk (VaR)
Maximum expected loss over a specific time period at a given confidence level.
- **95% VaR**: Expected maximum loss exceeded only 5% of the time
- **99% VaR**: Expected maximum loss exceeded only 1% of the time

### Conditional VaR (CVaR)
Average loss in the worst-case scenarios beyond the VaR threshold.
Also known as Expected Shortfall (ES).

### Sharpe Ratio
Risk-adjusted return metric: `(Return - Risk-free Rate) / Volatility`
- Higher is better
- Measures return per unit of risk

### Maximum Drawdown
Largest peak-to-trough decline in portfolio value.

## Performance Benchmarks

On a typical modern CPU:
- **VaR Calculation**: < 1ms for 1,000 data points
- **Monte Carlo (10,000 sims)**: ~2-5 seconds
- **
