"""Configuration primitives for BankGuard AI.

Importing this module is deliberately side-effect free: it does not read a
``.env`` file, create directories, or initialize Ollama/Pinecone clients.
Applications opt in by calling :func:`load_settings`.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping

from dotenv import load_dotenv

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[1]

DEFAULT_NAMESPACE = "mx-pld-ft"
DEFAULT_CHAT_MODEL = "qwen3.5:4b"
DEFAULT_OLLAMA_NUM_PREDICT = 900
DEFAULT_EMBEDDING_MODEL = "bge-large"
DEFAULT_RETRIEVER_K = 5
DEFAULT_RETRIEVAL_K = DEFAULT_RETRIEVER_K

DEFAULT_TRANSACTIONS_CSV = PROJECT_ROOT / "agents" / "data" / "bank_transactions.csv"
DEFAULT_PROFILES_CSV = PROJECT_ROOT / "agents" / "data" / "client_profiles.csv"
DEFAULT_OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DEFAULT_RISK_CONFIG = PACKAGE_DIR / "risk_config.json"


def _repository_path(value: str | Path | None, default: Path) -> Path:
    """Resolve a configured path and reject escapes outside the repository."""
    candidate = default if value in {None, ""} else Path(value)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate

    resolved = candidate.expanduser().resolve()
    try:
        resolved.relative_to(PROJECT_ROOT)
    except ValueError as error:
        raise ValueError(
            f"Configured path must stay inside the repository: {candidate}"
        ) from error
    return resolved


def _positive_int(value: str | int, *, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


@dataclass(frozen=True, slots=True)
class BankGuardSettings:
    """Runtime settings; all external clients are constructed elsewhere."""

    project_root: Path
    transactions_csv: Path
    profiles_csv: Path
    outputs_dir: Path
    charts_dir: Path
    alerts_dir: Path
    risk_config_path: Path
    index_name: str | None
    corpus_id: str = DEFAULT_NAMESPACE
    pinecone_namespace: str = DEFAULT_NAMESPACE
    chat_model: str = DEFAULT_CHAT_MODEL
    ollama_num_predict: int = DEFAULT_OLLAMA_NUM_PREDICT
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    retriever_k: int = DEFAULT_RETRIEVER_K

    @property
    def ollama_chat_model(self) -> str:
        """Compatibility alias used by the compliance-agent factory."""
        return self.chat_model

    @property
    def regulatory_retrieval_k(self) -> int:
        """Compatibility alias for the regulatory retriever configuration."""
        return self.retriever_k

    def as_dict(self) -> dict[str, str | int | None]:
        """Return a JSON-compatible representation without secret values."""
        return {
            "project_root": self.project_root.as_posix(),
            "transactions_csv": self.transactions_csv.as_posix(),
            "profiles_csv": self.profiles_csv.as_posix(),
            "outputs_dir": self.outputs_dir.as_posix(),
            "charts_dir": self.charts_dir.as_posix(),
            "alerts_dir": self.alerts_dir.as_posix(),
            "risk_config_path": self.risk_config_path.as_posix(),
            "index_name": self.index_name,
            "corpus_id": self.corpus_id,
            "pinecone_namespace": self.pinecone_namespace,
            "chat_model": self.chat_model,
            "ollama_num_predict": self.ollama_num_predict,
            "embedding_model": self.embedding_model,
            "retriever_k": self.retriever_k,
        }


@dataclass(frozen=True, slots=True)
class RiskEngineConfig:
    """Configurable prototype rules; none are official regulatory limits."""

    schema_version: int = 1
    weights: Mapping[str, float] = field(
        default_factory=lambda: {
            "historical_deviation": 0.40,
            "atypical_frequency": 0.25,
            "amount_concentration": 0.20,
            "profile_inconsistency": 0.15,
        }
    )
    profile_weights: Mapping[str, float] = field(
        default_factory=lambda: {
            "amount": 0.40,
            "channel": 0.20,
            "location": 0.20,
            "transaction_type": 0.10,
            "currency": 0.10,
        }
    )
    medium_threshold: float = 40.0
    high_threshold: float = 70.0
    history_lookback_days: int = 92
    minimum_analysis_transactions: int = 3
    minimum_history_transactions: int = 8
    minimum_history_days: int = 30
    concentration_window_days: int = 3
    outlier_mad_z: float = 3.5
    burst_minimum_transactions: int = 4
    burst_rate_multiplier: float = 3.0
    maximum_analysis_days: int = 366
    allowed_currencies: tuple[str, ...] = ("MXN", "USD")

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.schema_version < 1:
            raise ValueError("schema_version must be at least 1")
        _validate_weight_map(self.weights, "weights")
        _validate_weight_map(self.profile_weights, "profile_weights")
        expected = {
            "historical_deviation",
            "atypical_frequency",
            "amount_concentration",
            "profile_inconsistency",
        }
        if set(self.weights) != expected:
            raise ValueError(f"weights must contain exactly {sorted(expected)!r}")
        expected_profile = {
            "amount",
            "channel",
            "location",
            "transaction_type",
            "currency",
        }
        if set(self.profile_weights) != expected_profile:
            raise ValueError(
                "profile_weights must contain exactly " f"{sorted(expected_profile)!r}"
            )
        if not 0 <= self.medium_threshold < self.high_threshold <= 100:
            raise ValueError("Risk thresholds must satisfy 0 <= medium < high <= 100")
        for name in (
            "history_lookback_days",
            "minimum_analysis_transactions",
            "minimum_history_transactions",
            "minimum_history_days",
            "concentration_window_days",
            "burst_minimum_transactions",
            "maximum_analysis_days",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.outlier_mad_z <= 0 or self.burst_rate_multiplier <= 0:
            raise ValueError("Outlier and burst multipliers must be positive")
        if not self.allowed_currencies:
            raise ValueError("allowed_currencies cannot be empty")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "RiskEngineConfig":
        known = set(cls.__dataclass_fields__)
        unknown = set(payload) - known
        if unknown:
            raise ValueError(f"Unknown risk configuration keys: {sorted(unknown)!r}")
        values = dict(payload)
        if "allowed_currencies" in values:
            values["allowed_currencies"] = tuple(values["allowed_currencies"])
        return cls(**values)


RiskConfig = RiskEngineConfig


def _validate_weight_map(weights: Mapping[str, float], name: str) -> None:
    if not weights:
        raise ValueError(f"{name} cannot be empty")
    numeric = [float(value) for value in weights.values()]
    if any(not math.isfinite(value) or value < 0 for value in numeric):
        raise ValueError(f"{name} values must be finite and non-negative")
    if not math.isclose(sum(numeric), 1.0, abs_tol=1e-9):
        raise ValueError(f"{name} must sum to 1.0")


def load_settings(
    *,
    env_file: str | Path | None = None,
    load_env_file: bool = True,
    environ: Mapping[str, str] | None = None,
) -> BankGuardSettings:
    """Load settings on demand from environment variables.

    ``environ`` is injectable so unit tests need neither a real ``.env`` file
    nor process-global environment mutations.
    """
    if load_env_file:
        dotenv_path = _repository_path(env_file, PROJECT_ROOT / ".env")
        load_dotenv(dotenv_path=dotenv_path, override=False)

    source = os.environ if environ is None else environ
    outputs_dir = _repository_path(
        source.get("BANKGUARD_OUTPUTS_DIR"), DEFAULT_OUTPUTS_DIR
    )

    corpus_id = source.get("RAG_CORPUS_ID", DEFAULT_NAMESPACE).strip()
    namespace = source.get("PINECONE_NAMESPACE", corpus_id).strip()
    chat_model = source.get("OLLAMA_CHAT_MODEL", DEFAULT_CHAT_MODEL).strip()
    embedding_model = source.get("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL).strip()
    if not namespace or not corpus_id or not chat_model or not embedding_model:
        raise ValueError("Corpus, namespace, and Ollama model names cannot be empty")

    index_name = source.get("INDEX_NAME")
    if index_name is not None:
        index_name = index_name.strip() or None

    return BankGuardSettings(
        project_root=PROJECT_ROOT,
        transactions_csv=_repository_path(
            source.get("BANKGUARD_TRANSACTIONS_CSV"),
            DEFAULT_TRANSACTIONS_CSV,
        ),
        profiles_csv=_repository_path(
            source.get("BANKGUARD_PROFILES_CSV"), DEFAULT_PROFILES_CSV
        ),
        outputs_dir=outputs_dir,
        charts_dir=_repository_path(
            source.get("BANKGUARD_CHARTS_DIR"), outputs_dir / "charts"
        ),
        alerts_dir=_repository_path(
            source.get("BANKGUARD_ALERTS_DIR"), outputs_dir / "alerts"
        ),
        risk_config_path=_repository_path(
            source.get("BANKGUARD_RISK_CONFIG"), DEFAULT_RISK_CONFIG
        ),
        index_name=index_name,
        corpus_id=corpus_id,
        pinecone_namespace=namespace,
        chat_model=chat_model,
        ollama_num_predict=_positive_int(
            source.get("OLLAMA_NUM_PREDICT", DEFAULT_OLLAMA_NUM_PREDICT),
            name="OLLAMA_NUM_PREDICT",
        ),
        embedding_model=embedding_model,
        retriever_k=_positive_int(
            source.get(
                "BANKGUARD_RETRIEVER_K",
                source.get("RAG_RETRIEVAL_K", DEFAULT_RETRIEVER_K),
            ),
            name="BANKGUARD_RETRIEVER_K",
        ),
    )


def load_risk_config(path: str | Path | None = None) -> RiskEngineConfig:
    """Load and validate the deterministic scoring configuration."""
    config_path = _repository_path(path, DEFAULT_RISK_CONFIG)
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Risk configuration not found: {config_path}"
        ) from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid risk configuration JSON: {config_path}") from error

    try:
        config = RiskEngineConfig.from_mapping(payload)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Incomplete risk configuration: {config_path}") from error
    return config


__all__ = [
    "BankGuardSettings",
    "DEFAULT_CHAT_MODEL",
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_NAMESPACE",
    "DEFAULT_OLLAMA_NUM_PREDICT",
    "DEFAULT_OUTPUTS_DIR",
    "DEFAULT_PROFILES_CSV",
    "DEFAULT_RETRIEVER_K",
    "DEFAULT_RETRIEVAL_K",
    "DEFAULT_RISK_CONFIG",
    "DEFAULT_TRANSACTIONS_CSV",
    "PACKAGE_DIR",
    "PROJECT_ROOT",
    "RiskEngineConfig",
    "RiskConfig",
    "load_risk_config",
    "load_settings",
]
