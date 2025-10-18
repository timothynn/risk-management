"""
Risk Management System
A comprehensive risk analysis toolkit for financial portfolios
"""

from .risk_engine import (
    RiskManager,
    RiskMetrics,
    MonteCarloEngine,
    VaRCalculator,
    StressTester,
)

__version__ = "1.0.0"
__author__ = "Risk Management Team"
__all__ = [
    "RiskManager",
    "RiskMetrics",
    "MonteCarloEngine",
    "VaRCalculator",
    "StressTester",
]
