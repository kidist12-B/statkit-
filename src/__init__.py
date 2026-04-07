# statkit/src/__init__.py
# Package initialiser — surfaces the two primary modules at the top level.

from .stat_engine import DataAnalyzer
from .monte_carlo import run_server_simulation, crash_trial

__all__ = ["DataAnalyzer", "run_server_simulation", "crash_trial"]
