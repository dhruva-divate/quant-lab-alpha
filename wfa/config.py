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
    def __post_init__(self):
        self.validate()
    #exception handling for weird scenarios REQUEST CHECK FOR ALL VALIDATIONS
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
    # Performance metrics (MAY NEED TO ADD DEFAULTS)
    equity_curve: pd.Series  # Time series of portfolio value
    returns: pd.Series  # Time series of portfolio returns
    regime_labels: pd.Series  # Time series of detected regimes
    weights_history: pd.DataFrame  # Time series of portfolio weights
    total_return: float = 0.0  # Total return over the period
    annualised_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    turnover: float = 0.0
    transaction_costs: float = 0.0

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
    
@dataclass
class WFAResult:
    """Probably the most involved dataclass, stores the overall WFA results.
    from a design perspective, this is the one that will guide the rest of the development of this module."""
    config: WFAConfig
    #combined series from all windows
    complete_equity_curve: pd.Series
    full_returns: pd.Series
    full_regime_labels: pd.Series
    # per Window results
    in_sample_results: List[BacktestResult] = field(default_factory=list)
    out_sample_results: List[BacktestResult] = field(default_factory=list)
    #transition matrix of regimes (final HMM)
    transition_matrix: Optional[np.ndarray] = None
    regime_names: List[str]
    # buy and hold benchmark 
    benchmark_equity_curve: pd.Series
    benchmark_returns: pd.Series

    #aggregate metrics oos=out of sample
    oos_total_return: float = 0.0
    oos_annualised_return: float = 0.0
    oos_volatility: float = 0.0
    oos_sharpe_ratio: float = 0.0
    oos_max_drawdown: float = 0.0
    benchmark_total_return: float = 0.0 
    benchmark_sharpe_ratio: float = 0.0

    def compute_aggregate_metrics(self, risk_free_rate: float = 0.02):
        if len(self.out_sample_results) == 0:
            raise ValueError("No out-of-sample results to aggregate.")
        """
        if len(self.out_sample_results) == 0:
            return
        Can be implemented later if the we decide to handle the zero case, currently gives error to catch it
        """
        oos_returns = pd.concat([res.returns for res in self.out_sample_results])
        self.oos_total_return = (1 + oos_returns).prod() - 1 #total return
        n_months = len(oos_returns)
        self.oos_annualised_return = (1 + self.oos_total_return) ** (12 / n_months) - 1
        self.oos_volatility = oos_returns.std() * np.sqrt(12)  # annualised volatility

        # sharpe ratio calculation
        monthly_rf=risk_free_rate/12
        excess_return = self.oos_annualised_return - risk_free_rate
        if self.oos_volatility > 0:
            self.oos_sharpe_ratio = excess_return / self.oos_volatility
        else:
            self.oos_sharpe_ratio = np.nan #undefined sharpe ratio
        
        # max drawdown calculation
        equity= (1 + oos_returns).cumprod() 
        rolling_max = equity.cummax()
        drawdowns = (equity - rolling_max) / rolling_max
        self.oos_max_drawdown = drawdowns.min()  # most negative drawdown
        # Benchmark metrics MAY NEED TWEAKING TO ALIGN DATES
        if len(self.benchmark_returns) > 0:
            bench_aligned = self.benchmark_returns.loc[oos_returns.index]
            self.benchmark_total_return = (1 + bench_aligned).prod() - 1
            bench_vol = bench_aligned.std() * np.sqrt(12)
            bench_ann_ret = (1 + self.benchmark_total_return) ** (12 / len(bench_aligned)) - 1
            if bench_vol > 0:
                self.benchmark_sharpe_ratio = (bench_ann_ret - risk_free_rate) / bench_vol
        else:
            self.benchmark_total_return = np.nan
            self.benchmark_sharpe_ratio = np.nan
    def get_regime_statistics(self) -> Dict[str, Dict[str, float]]:
            stats={}
            for regime in self.regime_names:
                mask = self.full_regime_labels == regime
                regime_returns = self.full_returns[mask]
                if len(regime_returns) == 0:
                    continue
                stats[regime] = {
                    'count': len(regime_returns),
                    'mean_return': regime_returns.mean() *12 ,  # Annualised
                    'volatility': regime_returns.std() * np.sqrt(12),  # Annualised
                    'hit_rate': (regime_returns > 0).mean()
                }
            return stats
        
# policy presets per feature request and google
# Policy presets
POLICY_PRESETS = {
    'Defensive': [
        PolicyRule('Bull', {'Mkt-RF': 0.1, 'SMB': 0.05}, 'Moderate market exposure'),
        PolicyRule('Bear', {'HML': 0.15, 'RMW': 0.2, 'CMA': 0.1}, 'Quality/Value tilt'),
        PolicyRule('Volatile', {'RMW': 0.25, 'CMA': 0.15}, 'High quality tilt'),
    ],
    'Momentum': [
        PolicyRule('Bull', {'Mkt-RF': 0.2, 'SMB': 0.1}, 'Aggressive growth'),
        PolicyRule('Bear', {'Mkt-RF': -0.3}, 'Reduce market exposure'),
        PolicyRule('Volatile', {}, 'No adjustment'),
    ],
    'Contrarian': [
        PolicyRule('Bull', {'Mkt-RF': -0.1, 'HML': 0.1}, 'Reduce exposure, add value'),
        PolicyRule('Bear', {'Mkt-RF': 0.15, 'SMB': 0.1}, 'Add exposure on weakness'),
        PolicyRule('Volatile', {'Mkt-RF': 0.1}, 'Moderate increase'),
    ],
}



