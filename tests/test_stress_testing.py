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
