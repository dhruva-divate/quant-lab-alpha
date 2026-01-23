# Dataclass decortator is used to create classes that primarily store data

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

@dataclass
class RegimeConfig:
    """Configuration for HMM regime detection.

       Description of regimes:

       Bull market: High returns, low volatility
       Bear market: Negative returns, high volatility
       Volatile market: High volatility, mixed returns

       This defines lookback periods (in months) for calculating market features:

       'return_1m': 1 → 1-month momentum
       'return_3m': 3 → 3-month momentum
       'return_12m': 12 → 12-month momentum (trend strength)
       'vol_3m': 3 → 3-month volatility (short-term risk)
       'vol_12m': 12 → 12-month volatility (long-term risk)
       'beta_window': 36 → 36-month beta calculation (market sensitivity)
    """
    n_regimes: int = 3
    covariance_type: str = 'full' #full covariance matrix
    n_iter: int = 100
    random_state: int = 69
    # Feature extraction parameters
    rolling_windows: Dict[str, int] = field(default_factory=lambda: { #using field() to make a new dict isntance each time
        'return_1m': 1,
        'return_3m': 3,
        'return_12m': 12,
        'vol_3m': 3,
        'vol_12m': 12,
        'beta_window': 36
    })
@dataclass
class PolicyRule:
    """Defines a regime-specific policy rule with factor tilts."""
    regime: str  # 'Bull', 'Bear', 'Volatile'
    factor_tilts: Dict[str, float]  # supposed to be factor: exposure
    description: str = "" # Optional description of the rule LATER

    def __post_init__(self):
        valid_regimes = {'Bull', 'Bear', 'Volatile'}
        if self.regime not in valid_regimes:
            raise ValueError(f"Regime must be one of {valid_regimes}, got '{self.regime}'")
@dataclass
class WFAConfig:
    # Main configuration for Walk-Forward Analysis.
    # Window parameters (in months)
    in_sample_months: int = 60  # 5 years
    out_sample_months: int = 12  # 1 year
    step_months: int = 12  # 1 year step
    # Regime detection
    regime_config: RegimeConfig = field(default_factory=RegimeConfig) #for proper handling of instances
    # Transaction costs defaults in basis points
    transaction_cost_bps: float = 10.0  
    slippage_bps: float = 5.0
    # Policy preset name
    policy_preset: str = 'Defensive'  # 'Defensive', 'Momentum', 'Contrarian', 'Custom'
    # Custom policy rules (used when policy_preset='Custom')
    custom_rules: List[PolicyRule] = field(default_factory=list)

    #exception handling for weird scenarios REQUEST CHECK FOR ALL VALIDATIONSx
    def validate(self):
        if self.in_sample_months < 24:
            raise ValueError("In-sample window must be at least 24 months")
        if self.out_sample_months < 1:
            raise ValueError("Out-of-sample window must be at least 1 month")
        if self.step_months < 1:
            raise ValueError("Step size must be at least 1 month")
        if self.transaction_cost_bps < 0:
            raise ValueError("Transaction cost cannot be negative")
        if self.slippage_bps < 0:
            raise ValueError("Slippage cannot be negative")
        if self.policy_preset == 'Custom' and not self.custom_rules:
            raise ValueError("Custom rules must be provided when policy_preset is 'Custom'")
        # Additional validations can be added as needed

@dataclass
class BacktestResult:
    """Stores the results of a single backtest."""
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    is_in_sample: bool
    # Performance metrics
    equity_curve: pd.Series  # Time series of portfolio value
    returns: pd.Series  # Time series of portfolio returns
    regime_labels: pd.Series  # Time series of detected regimes
    weights_history: pd.DataFrame  # Time series of portfolio weights
    total_return: float # default 0.0? HUSAIN
    annualised_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    turnover: float
    transaction_costs: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert the backtest result to a dictionary."""
        return {
            'start_date': self.start_date,
            'end_date': self.end_date,
            'is_in_sample': self.is_in_sample,
            'total_return': self.total_return,
            'annualised_return': self.annualised_return,
            'volatility': self.volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'turnover': self.turnover,
            'transaction_costs': self.transaction_costs
        }




