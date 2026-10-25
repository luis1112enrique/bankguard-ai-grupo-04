"""BankGuard AI backend package.

The package exports only configuration and data contracts at import time.
External services remain lazily initialized by their owning modules.
"""

from typing import Any

from .config import (
    BankGuardSettings,
    RiskConfig,
    RiskEngineConfig,
    load_risk_config,
    load_settings,
)
from .schemas import (
    AnalysisRequest,
    AnalysisResult,
    AnalysisTrace,
    ChartArtifact,
    ComplianceResult,
    RegulatoryEvidence,
    ReviewStatus,
    RiskAlert,
    RiskFactor,
    RiskFactors,
    RiskLevel,
    RiskScore,
    TraceEvent,
    TransactionMetrics,
)

__version__ = "0.1.0"


def run_analysis(*args: Any, **kwargs: Any) -> AnalysisResult:
    """Lazily call the UI-neutral BankGuard backend entry point."""
    from .orchestrator import run_analysis as _run_analysis

    return _run_analysis(*args, **kwargs)


__all__ = [
    "AnalysisRequest",
    "AnalysisResult",
    "AnalysisTrace",
    "BankGuardSettings",
    "ChartArtifact",
    "ComplianceResult",
    "RegulatoryEvidence",
    "ReviewStatus",
    "RiskAlert",
    "RiskConfig",
    "RiskEngineConfig",
    "RiskFactor",
    "RiskFactors",
    "RiskLevel",
    "RiskScore",
    "TraceEvent",
    "TransactionMetrics",
    "load_risk_config",
    "load_settings",
    "run_analysis",
]
