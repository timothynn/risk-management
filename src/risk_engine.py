"""
Risk Management Engine
Implements VaR, CVaR, Monte Carlo simulation, and stress testing
"""
import numpy as np
from numba import jit
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import pandas as pd


@dataclass
class RiskMetrics:
    """Container for risk metrics"""
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float


class MonteCarloEngine:
    """High-performance Monte Carlo simulation engine"""
    
    def __init__(self, num_simulations: int = 10000, seed: Optional[int] = None):
        self.num_simulations = num_simulations
        if seed:
            np.random.seed(seed)
    
    @staticmethod
    @jit(nopython=True)
    def _simulate_gbm(S0: float, mu: float, sigma: float, T: float, 
                      steps: int, paths: int) -> np.ndarray:
        """
        Geometric Brownian Motion simulation (JIT compiled)
        S0: Initial price
        mu: Expected return
        sigma: Volatility
        T: Time horizon
        steps: Number of time steps
        paths: Number of simulation paths
        """
        dt = T / steps
        prices = np.zeros((paths, steps + 1))
        prices[:, 0] = S0
        
        for i in range(paths):
            for t in range(1, steps + 1):
                z = np.random.standard_normal()
                prices[i, t] = prices[i, t-1] * np.exp(
                    (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
                )
        
        return prices
    
    def simulate_portfolio(self, 
                          initial_value: float,
                          weights: np.ndarray,
                          returns: np.ndarray,
                          covariance: np.ndarray,
                          time_horizon: int = 252) -> np.ndarray:
        """
        Simulate portfolio returns using multivariate normal distribution
        """
        portfolio_returns = np.random.multivariate_normal(
            returns, 
            covariance, 
            size=(self.num_simulations, time_horizon)
        )
        
        # Calculate portfolio value evolution
        portfolio_values = np.zeros((self.num_simulations, time_horizon + 1))
        portfolio_values[:, 0] = initial_value
        
        for i in range(time_horizon):
            period_return = np.dot(portfolio_returns[:, i], weights)
            portfolio_values[:, i + 1] = portfolio_values[:, i] * (1 + period_return)
        
        return portfolio_values


class VaRCalculator:
    """Value at Risk calculation using multiple methods"""
    
    @staticmethod
    def historical_var(returns: np.ndarray, confidence_level: float = 0.95) -> float:
        """Calculate VaR using historical method"""
        return np.percentile(returns, (1 - confidence_level) * 100)
    
    @staticmethod
    def parametric_var(returns: np.ndarray, confidence_level: float = 0.95) -> float:
        """Calculate VaR using parametric (variance-covariance) method"""
        mean = np.mean(returns)
        std = np.std(returns)
        z_score = {0.90: 1.28, 0.95: 1.645, 0.99: 2.326}[confidence_level]
        return mean - z_score * std
    
    @staticmethod
    def monte_carlo_var(simulated_returns: np.ndarray, 
                       confidence_level: float = 0.95) -> float:
        """Calculate VaR using Monte Carlo simulation"""
        return np.percentile(simulated_returns, (1 - confidence_level) * 100)
    
    @staticmethod
    def conditional_var(returns: np.ndarray, confidence_level: float = 0.95) -> float:
        """Calculate CVaR (Expected Shortfall)"""
        var = VaRCalculator.historical_var(returns, confidence_level)
        return np.mean(returns[returns <= var])


class StressTester:
    """Stress testing engine for portfolio analysis"""
    
    def __init__(self):
        self.scenarios = {}
    
    def add_scenario(self, name: str, shock_factors: Dict[str, float]):
        """Add a stress test scenario"""
        self.scenarios[name] = shock_factors
    
    def apply_scenario(self, 
                       portfolio_value: float,
                       positions: Dict[str, float],
                       scenario_name: str) -> float:
        """Apply stress scenario to portfolio"""
        if scenario_name not in self.scenarios:
            raise ValueError(f"Scenario {scenario_name} not found")
        
        shocked_value = portfolio_value
        shock_factors = self.scenarios[scenario_name]
        
        for asset, position_value in positions.items():
            if asset in shock_factors:
                shock = shock_factors[asset]
                shocked_value += position_value * shock
        
        return shocked_value
    
    def create_market_crash_scenario(self, severity: float = -0.20):
        """Create a market crash stress scenario"""
        self.add_scenario("market_crash", {
            "equities": severity,
            "corporate_bonds": severity * 0.5,
            "real_estate": severity * 0.7,
            "commodities": severity * 0.3
        })
    
    def create_rate_shock_scenario(self, bp_change: float = 200):
        """Create interest rate shock scenario (basis points)"""
        rate_change = bp_change / 10000
        self.add_scenario("rate_shock", {
            "bonds": -rate_change * 5,  # Duration approximation
            "real_estate": -rate_change * 3,
            "equities": -rate_change * 2
        })


class RiskManager:
    """Main risk management system"""
    
    def __init__(self, portfolio_value: float):
        self.portfolio_value = portfolio_value
        self.mc_engine = MonteCarloEngine()
        self.var_calc = VaRCalculator()
        self.stress_tester = StressTester()
        self.positions: Dict[str, float] = {}
        
    def add_position(self, asset: str, value: float):
        """Add position to portfolio"""
        self.positions[asset] = value
    
    def calculate_risk_metrics(self, 
                              returns: np.ndarray,
                              risk_free_rate: float = 0.02) -> RiskMetrics:
        """Calculate comprehensive risk metrics"""
        
        # VaR and CVaR
        var_95 = self.var_calc.historical_var(returns, 0.95)
        var_99 = self.var_calc.historical_var(returns, 0.99)
        cvar_95 = self.var_calc.conditional_var(returns, 0.95)
        cvar_99 = self.var_calc.conditional_var(returns, 0.99)
        
        # Volatility
        volatility = np.std(returns) * np.sqrt(252)
        
        # Sharpe Ratio
        mean_return = np.mean(returns) * 252
        sharpe = (mean_return - risk_free_rate) / volatility if volatility > 0 else 0
        
        # Maximum Drawdown
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        return RiskMetrics(
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            volatility=volatility,
            sharpe_ratio=sharpe,
            max_drawdown=max_drawdown
        )
    
    def run_monte_carlo_analysis(self,
                                 weights: np.ndarray,
                                 expected_returns: np.ndarray,
                                 covariance: np.ndarray,
                                 time_horizon: int = 252) -> Tuple[np.ndarray, Dict]:
        """Run complete Monte Carlo analysis"""
        
        simulations = self.mc_engine.simulate_portfolio(
            self.portfolio_value,
            weights,
            expected_returns,
            covariance,
            time_horizon
        )
        
        # Calculate final returns
        final_values = simulations[:, -1]
        returns = (final_values - self.portfolio_value) / self.portfolio_value
        
        # Statistics
        stats = {
            'mean_final_value': np.mean(final_values),
            'median_final_value': np.median(final_values),
            'std_final_value': np.std(final_values),
            'var_95': self.var_calc.monte_carlo_var(returns, 0.95),
            'var_99': self.var_calc.monte_carlo_var(returns, 0.99),
            'prob_loss': np.sum(returns < 0) / len(returns),
            'best_case': np.max(returns),
            'worst_case': np.min(returns)
        }
        
        return simulations, stats
    
    def run_stress_tests(self) -> Dict[str, float]:
        """Run all defined stress test scenarios"""
        results = {}
        
        # Add default scenarios
        self.stress_tester.create_market_crash_scenario(-0.30)
        self.stress_tester.create_rate_shock_scenario(300)
        
        for scenario_name in self.stress_tester.scenarios.keys():
            stressed_value = self.stress_tester.apply_scenario(
                self.portfolio_value,
                self.positions,
                scenario_name
            )
            pnl = stressed_value - self.portfolio_value
            pnl_pct = (pnl / self.portfolio_value) * 100
            results[scenario_name] = pnl_pct
        
        return results
    
    def generate_risk_report(self, returns: np.ndarray) -> str:
        """Generate comprehensive risk report"""
        metrics = self.calculate_risk_metrics(returns)
        
        report = f"""
{'='*60}
RISK MANAGEMENT REPORT
{'='*60}

Portfolio Value: ${self.portfolio_value:,.2f}

VALUE AT RISK (VaR)
-------------------
95% VaR (1-day): {metrics.var_95*100:.2f}%  (${self.portfolio_value * metrics.var_95:,.2f})
99% VaR (1-day): {metrics.var_99*100:.2f}%  (${self.portfolio_value * metrics.var_99:,.2f})

CONDITIONAL VALUE AT RISK (CVaR)
---------------------------------
95% CVaR: {metrics.cvar_95*100:.2f}%  (${self.portfolio_value * metrics.cvar_95:,.2f})
99% CVaR: {metrics.cvar_99*100:.2f}%  (${self.portfolio_value * metrics.cvar_99:,.2f})

RISK METRICS
------------
Annualized Volatility: {metrics.volatility*100:.2f}%
Sharpe Ratio: {metrics.sharpe_ratio:.2f}
Maximum Drawdown: {metrics.max_drawdown*100:.2f}%

{'='*60}
"""
        return report
