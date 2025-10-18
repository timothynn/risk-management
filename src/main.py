"""
Risk Management System - Main Application
Demonstrates VaR, Monte Carlo, and Stress Testing capabilities
"""
import numpy as np
import pandas as pd
from risk_engine import RiskManager, MonteCarloEngine
import time


def generate_sample_data(n_days: int = 1000, n_assets: int = 4):
    """Generate sample market data for demonstration"""
    # Simulate daily returns for multiple assets
    np.random.seed(42)
    
    # Asset characteristics
    mean_returns = np.array([0.0008, 0.0006, 0.0004, 0.0005])  # Daily
    volatilities = np.array([0.02, 0.015, 0.01, 0.012])  # Daily
    
    # Correlation matrix
    correlation = np.array([
        [1.0, 0.7, 0.5, 0.6],
        [0.7, 1.0, 0.6, 0.5],
        [0.5, 0.6, 1.0, 0.4],
        [0.6, 0.5, 0.4, 1.0]
    ])
    
    # Covariance matrix
    cov_matrix = np.outer(volatilities, volatilities) * correlation
    
    # Generate returns
    returns = np.random.multivariate_normal(mean_returns, cov_matrix, n_days)
    
    # Create DataFrame
    df = pd.DataFrame(
        returns,
        columns=['Equities', 'Bonds', 'Real_Estate', 'Commodities']
    )
    
    return df, mean_returns, cov_matrix


def example_basic_var_analysis():
    """Example: Basic VaR analysis on historical data"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Historical VaR Analysis")
    print("="*70)
    
    # Generate sample data
    returns_df, _, _ = generate_sample_data(1000)
    portfolio_returns = returns_df['Equities'].values
    
    # Initialize risk manager
    initial_value = 1_000_000
    risk_mgr = RiskManager(initial_value)
    
    # Calculate risk metrics
    start_time = time.time()
    metrics = risk_mgr.calculate_risk_metrics(portfolio_returns)
    calc_time = time.time() - start_time
    
    print(f"\nPortfolio Value: ${initial_value:,.2f}")
    print(f"Analysis Period: {len(portfolio_returns)} days")
    print(f"\nRisk Metrics:")
    print(f"  95% VaR: {metrics.var_95*100:.3f}% (${initial_value * abs(metrics.var_95):,.2f})")
    print(f"  99% VaR: {metrics.var_99*100:.3f}% (${initial_value * abs(metrics.var_99):,.2f})")
    print(f"  95% CVaR: {metrics.cvar_95*100:.3f}% (${initial_value * abs(metrics.cvar_95):,.2f})")
    print(f"  99% CVaR: {metrics.cvar_99*100:.3f}% (${initial_value * abs(metrics.cvar_99):,.2f})")
    print(f"  Volatility (Annual): {metrics.volatility*100:.2f}%")
    print(f"  Sharpe Ratio: {metrics.sharpe_ratio:.3f}")
    print(f"  Max Drawdown: {metrics.max_drawdown*100:.2f}%")
    print(f"\nCalculation time: {calc_time*1000:.2f}ms")


def example_monte_carlo_simulation():
    """Example: Monte Carlo portfolio simulation"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Monte Carlo Portfolio Simulation")
    print("="*70)
    
    # Portfolio setup
    initial_value = 1_000_000
    weights = np.array([0.4, 0.3, 0.2, 0.1])  # Asset allocation
    
    # Generate market parameters
    _, expected_returns, cov_matrix = generate_sample_data()
    
    # Initialize risk manager
    risk_mgr = RiskManager(initial_value)
    
    # Run Monte Carlo simulation
    print(f"\nRunning {risk_mgr.mc_engine.num_simulations:,} simulations...")
    print(f"Time horizon: 252 days (1 year)")
    print(f"Portfolio allocation: Equities {weights[0]*100:.0f}%, Bonds {weights[1]*100:.0f}%, "
          f"RE {weights[2]*100:.0f}%, Commodities {weights[3]*100:.0f}%")
    
    start_time = time.time()
    simulations, stats = risk_mgr.run_monte_carlo_analysis(
        weights, expected_returns, cov_matrix, time_horizon=252
    )
    calc_time = time.time() - start_time
    
    print(f"\nMonte Carlo Results:")
    print(f"  Mean Final Value: ${stats['mean_final_value']:,.2f}")
    print(f"  Median Final Value: ${stats['median_final_value']:,.2f}")
    print(f"  Std Dev: ${stats['std_final_value']:,.2f}")
    print(f"  Best Case Return: {stats['best_case']*100:.2f}%")
    print(f"  Worst Case Return: {stats['worst_case']*100:.2f}%")
    print(f"  Probability of Loss: {stats['prob_loss']*100:.2f}%")
    print(f"  95% VaR: {stats['var_95']*100:.3f}%")
    print(f"  99% VaR: {stats['var_99']*100:.3f}%")
    print(f"\nSimulation time: {calc_time:.2f}s ({risk_mgr.mc_engine.num_simulations/calc_time:.0f} simulations/sec)")


def example_stress_testing():
    """Example: Stress testing analysis"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Stress Testing")
    print("="*70)
    
    # Portfolio setup
    initial_value = 1_000_000
    risk_mgr = RiskManager(initial_value)
    
    # Define positions
    positions = {
        'equities': 400_000,
        'corporate_bonds': 300_000,
        'real_estate': 200_000,
        'commodities': 100_000
    }
    
    for asset, value in positions.items():
        risk_mgr.add_position(asset, value)
    
    print(f"\nPortfolio Composition:")
    for asset, value in positions.items():
        print(f"  {asset.title()}: ${value:,.2f} ({value/initial_value*100:.1f}%)")
    
    # Run stress tests
    print(f"\nRunning stress test scenarios...")
    start_time = time.time()
    stress_results = risk_mgr.run_stress_tests()
    calc_time = time.time() - start_time
    
    print(f"\nStress Test Results:")
    for scenario, pnl_pct in stress_results.items():
        pnl_dollar = (pnl_pct / 100) * initial_value
        print(f"  {scenario.replace('_', ' ').title()}: {pnl_pct:+.2f}% (${pnl_dollar:+,.2f})")
    
    print(f"\nCalculation time: {calc_time*1000:.2f}ms")


def example_full_risk_report():
    """Example: Generate comprehensive risk report"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Comprehensive Risk Report")
    print("="*70)
    
    # Generate portfolio data
    returns_df, _, _ = generate_sample_data(1000)
    weights = np.array([0.4, 0.3, 0.2, 0.1])
    portfolio_returns = returns_df.values @ weights
    
    # Initialize risk manager
    initial_value = 1_000_000
    risk_mgr = RiskManager(initial_value)
    
    # Generate report
    report = risk_mgr.generate_risk_report(portfolio_returns)
    print(report)


def benchmark_performance():
    """Benchmark system performance"""
    print("\n" + "="*70)
    print("PERFORMANCE BENCHMARK")
    print("="*70)
    
    sizes = [1_000, 10_000, 50_000, 100_000]
    
    print("\nMonte Carlo Simulation Performance:")
    print(f"{'Simulations':<15} {'Time (s)':<12} {'Sims/sec':<12}")
    print("-" * 40)
    
    for n_sims in sizes:
        risk_mgr = RiskManager(1_000_000)
        risk_mgr.mc_engine = MonteCarloEngine(num_simulations=n_sims, seed=42)
        
        _, expected_returns, cov_matrix = generate_sample_data()
        weights = np.array([0.4, 0.3, 0.2, 0.1])
        
        start_time = time.time()
        _ = risk_mgr.run_monte_carlo_analysis(
            weights, expected_returns, cov_matrix, time_horizon=252
        )
        calc_time = time.time() - start_time
        
        throughput = n_sims / calc_time
        print(f"{n_sims:<15,} {calc_time:<12.3f} {throughput:<12,.0f}")


def main():
    """Main application entry point"""
    print("\n" + "="*70)
    print("RISK MANAGEMENT SYSTEM")
    print("High-Performance Risk Analysis with VaR, Monte Carlo & Stress Testing")
    print("="*70)
    
    # Run examples
    example_basic_var_analysis()
    example_monte_carlo_simulation()
    example_stress_testing()
    example_full_risk_report()
    benchmark_performance()
    
    print("\n" + "="*70)
    print("Analysis Complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
