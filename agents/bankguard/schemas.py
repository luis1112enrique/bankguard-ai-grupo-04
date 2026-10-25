"""Serializable data contracts shared by the BankGuard agents and API."""

from __future__ import annotations

from datetime import date, datetime, timezone
from enum import Enum
from typing import Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
    model_validator,
)


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


class SerializableModel(BaseModel):
    """Strict base model with explicit JSON-safe serialization helpers."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    def to_dict(self) -> dict[str, JsonValue]:
        return self.model_dump(mode="json")

    def to_json(self, *, indent: int | None = None) -> str:
        return self.model_dump_json(indent=indent)


class RiskLevel(str, Enum):
    LOW = "bajo"
    MEDIUM = "medio"
    HIGH = "alto"
    INCONCLUSIVE = "no_concluyente"


class FactorName(str, Enum):
    DEVIATION = "historical_deviation"
    ATYPICAL_FREQUENCY = "atypical_frequency"
    CONCENTRATION = "amount_concentration"
    PROFILE_INCONSISTENCY = "profile_inconsistency"


class BindingStatus(str, Enum):
    BINDING = "binding"
    GUIDANCE = "guidance"
    REFERENCE = "reference"
    UNKNOWN = "unknown"


class ComplianceStatus(str, Enum):
    SUPPORTED = "supported"
    NOT_SUPPORTED = "not_supported"
    REQUIRES_REVIEW = "requires_review"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    NOT_REQUESTED = "not_requested"
    ERROR = "error"


class ExecutionStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    INCONCLUSIVE = "inconclusive"
    ERROR = "error"


class TraceStatus(str, Enum):
    STARTED = "started"
    SUCCESS = "success"
    SKIPPED = "skipped"
    ERROR = "error"


class ChartType(str, Enum):
    TRANSACTION_EVOLUTION = "transaction_evolution"
    RISK_DISTRIBUTION = "risk_distribution"


class ReviewStatus(str, Enum):
    PENDING = "pending"
    NOT_TRIGGERED = "not_triggered"


class AnalysisRequest(SerializableModel):
    """Validated request accepted by the public ``run_analysis`` API."""

    client_id: str = Field(min_length=1, max_length=64)
    start_date: date
    end_date: date
    compliance_question: str | None = Field(
        default=None,
        min_length=3,
        max_length=2_000,
    )
    include_compliance: bool = True
    include_charts: bool = True
    generate_alert: bool = True

    @field_validator("client_id")
    @classmethod
    def validate_client_id(cls, value: str) -> str:
        normalized = value.strip().upper()
        if not normalized.replace("-", "").replace("_", "").isalnum():
            raise ValueError(
                "client_id may contain only letters, numbers, hyphens, and underscores"
            )
        return normalized

    @field_validator("compliance_question", mode="before")
    @classmethod
    def normalize_question(cls, value: object) -> object:
        if value is None:
            return None
        if not isinstance(value, str):
            return value
        normalized = value.strip()
        return normalized or None

    @model_validator(mode="after")
    def validate_date_range(self) -> Self:
        if self.end_date < self.start_date:
            raise ValueError("end_date must be on or after start_date")
        return self


class TransactionMetrics(SerializableModel):
    """Observed and historical aggregates for one client and date range."""

    client_id: str
    start_date: date
    end_date: date
    currency: str | None = Field(default=None, pattern=r"^[A-Z]{3}$")
    transaction_count: int = Field(ge=0)
    total_amount: float = Field(ge=0)
    period_days: int = Field(ge=1)
    average_amount: float | None = Field(default=None, ge=0)
    maximum_amount: float | None = Field(default=None, ge=0)
    standard_deviation_amount: float | None = Field(default=None, ge=0)
    active_days: int = Field(ge=0)
    average_daily_frequency: float = Field(ge=0)
    average_weekly_frequency: float = Field(ge=0)
    atypical_transaction_count: int = Field(default=0, ge=0)
    atypical_transaction_rate_percent: float | None = Field(default=None, ge=0, le=100)
    amount_hhi: float | None = Field(default=None, ge=0, le=1)
    normalized_hhi: float | None = Field(default=None, ge=0, le=100)
    rolling_window_amount_share_percent: float | None = Field(
        default=None, ge=0, le=100
    )
    historical_transaction_count: int = Field(default=0, ge=0)
    historical_coverage_days: int = Field(default=0, ge=0)
    historical_total_amount: float = Field(default=0, ge=0)
    historical_average_amount: float | None = Field(default=None, ge=0)
    expected_amount_from_history: float | None = Field(default=None, ge=0)
    expected_transaction_count_from_history: float | None = Field(default=None, ge=0)
    amount_deviation_percent: float | None = None
    frequency_deviation_percent: float | None = None
    current_data_sufficient: bool
    historical_data_sufficient: bool
    atypical_transaction_ids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class RiskFactor(SerializableModel):
    """Normalized factor score and its contribution to the total score."""

    name: FactorName
    normalized_score: float | None = Field(default=None, ge=0, le=100)
    weight: float = Field(ge=0, le=100)
    weighted_points: float | None = Field(default=None, ge=0, le=100)
    observed_value: float | None = None
    baseline_value: float | None = None
    explanation: str = Field(min_length=1)
    details: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_weighted_points(self) -> Self:
        if (
            self.weighted_points is not None
            and self.weighted_points > self.weight + 1e-6
        ):
            raise ValueError("weighted_points cannot exceed the factor weight")
        return self


class RiskFactors(SerializableModel):
    historical_deviation: RiskFactor
    atypical_frequency: RiskFactor
    amount_concentration: RiskFactor
    profile_inconsistency: RiskFactor

    @model_validator(mode="after")
    def validate_names(self) -> Self:
        expected = {
            "historical_deviation": FactorName.DEVIATION,
            "atypical_frequency": FactorName.ATYPICAL_FREQUENCY,
            "amount_concentration": FactorName.CONCENTRATION,
            "profile_inconsistency": FactorName.PROFILE_INCONSISTENCY,
        }
        for field_name, factor_name in expected.items():
            if getattr(self, field_name).name != factor_name:
                raise ValueError(
                    f"{field_name} must use factor name {factor_name.value!r}"
                )
        return self


class ScoreThresholds(SerializableModel):
    medium: float = Field(default=40, ge=0, le=100)
    high: float = Field(default=70, ge=0, le=100)

    @model_validator(mode="after")
    def validate_order(self) -> Self:
        if self.medium >= self.high:
            raise ValueError("medium threshold must be lower than high threshold")
        return self


class RiskScore(SerializableModel):
    """Deterministic score; inconclusive results must never imply a risk level."""

    score: float | None = Field(default=None, ge=0, le=100)
    level: RiskLevel
    conclusive: bool
    factors: RiskFactors
    thresholds: ScoreThresholds = Field(default_factory=ScoreThresholds)
    rationale: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    model_name: str = "bankguard-deterministic-risk-v1"

    @model_validator(mode="after")
    def validate_conclusion(self) -> Self:
        if self.conclusive and self.level == RiskLevel.INCONCLUSIVE:
            raise ValueError("A conclusive score cannot use level 'inconclusive'")
        if self.conclusive and self.score is None:
            raise ValueError("A conclusive result requires a score")
        if not self.conclusive and self.level != RiskLevel.INCONCLUSIVE:
            raise ValueError("An inconclusive score must use level 'inconclusive'")
        if not self.conclusive and self.score is not None:
            raise ValueError("An inconclusive result must not assign a numeric score")
        return self


class RegulatoryEvidence(SerializableModel):
    """One retrieved passage with enough metadata for an auditable citation."""

    source: str = Field(min_length=1)
    citation: str = Field(min_length=1)
    excerpt: str = Field(min_length=1)
    title: str | None = None
    document_type: str | None = None
    authority: str | None = None
    jurisdiction: str | None = None
    page: int | None = Field(default=None, ge=1)
    article: str | None = None
    provision: str | None = None
    binding_status: BindingStatus = BindingStatus.UNKNOWN
    relevance_score: float | None = Field(default=None, ge=-1, le=1)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class ComplianceResult(SerializableModel):
    """Answer produced independently by the compliance specialist."""

    question: str | None = None
    status: ComplianceStatus
    answer: str
    evidence: list[RegulatoryEvidence] = Field(default_factory=list)
    binding_sources_used: int = Field(
        default=0,
        ge=0,
        description=(
            "Fragmentos vinculantes recuperados y disponibles para el agente; "
            "no implica que todos fueran citados en la redacción final."
        ),
    )
    guidance_sources_used: int = Field(
        default=0,
        ge=0,
        description=(
            "Fragmentos orientativos recuperados y disponibles para el agente; "
            "no implica que todos fueran citados en la redacción final."
        ),
    )
    human_review_required: bool = False
    caveats: list[str] = Field(default_factory=list)
    retrieval_query: str | None = None
    current_as_of: date | None = None

    @model_validator(mode="after")
    def validate_supported_evidence(self) -> Self:
        if self.status == ComplianceStatus.SUPPORTED and not self.evidence:
            raise ValueError("supported compliance requires retrieved evidence")
        return self


class TraceEvent(SerializableModel):
    """Sanitized trace record; summaries should never contain credentials."""

    sequence: int = Field(ge=0)
    component: str = Field(min_length=1)
    action: str = Field(min_length=1)
    status: TraceStatus
    started_at: datetime = Field(default_factory=utc_now)
    completed_at: datetime | None = None
    elapsed_ms: float | None = Field(default=None, ge=0)
    input_summary: str | None = None
    output_summary: str | None = None
    error: str | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class AnalysisTrace(SerializableModel):
    run_id: str = Field(min_length=1)
    started_at: datetime = Field(default_factory=utc_now)
    completed_at: datetime | None = None
    events: list[TraceEvent] = Field(default_factory=list)


class ChartArtifact(SerializableModel):
    chart_type: ChartType
    relative_path: str = Field(min_length=1)
    title: str = Field(min_length=1)
    mime_type: str = "image/png"
    created_at: datetime = Field(default_factory=utc_now)
    description: str | None = None
    sha256: str | None = Field(default=None, pattern=r"^[a-fA-F0-9]{64}$")
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class RiskAlert(SerializableModel):
    """Structured analytical signal; it never alleges criminal conduct."""

    model_config = ConfigDict(frozen=True)

    alert_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    client_id: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=utc_now)
    period_start: date
    period_end: date
    triggered: bool
    review_status: ReviewStatus
    score: float | None = Field(default=None, ge=0, le=100)
    level: RiskLevel
    title: str = Field(min_length=1)
    message: str = Field(min_length=1)
    transaction_summary: dict[str, JsonValue] = Field(default_factory=dict)
    indicators: dict[str, JsonValue] = Field(default_factory=dict)
    contributing_factors: list[str] = Field(default_factory=list)
    atypical_operations: list[dict[str, JsonValue]] = Field(default_factory=list)
    compliance: ComplianceResult | None = None
    normative_references: list[RegulatoryEvidence] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)
    artifact_relative_path: str | None = None
    disclaimer: str = (
        "Resultado analítico simulado para revisión humana; no constituye una "
        "acusación, determinación legal ni reporte regulatorio."
    )

    @model_validator(mode="after")
    def validate_alert_state(self) -> Self:
        if self.period_end < self.period_start:
            raise ValueError("period_end must be on or after period_start")
        expected_trigger = self.level in {
            RiskLevel.MEDIUM,
            RiskLevel.HIGH,
            RiskLevel.INCONCLUSIVE,
        }
        if self.triggered != expected_trigger:
            raise ValueError("triggered is inconsistent with the risk level")
        expected_review = (
            ReviewStatus.PENDING if self.triggered else ReviewStatus.NOT_TRIGGERED
        )
        if self.review_status != expected_review:
            raise ValueError("review_status is inconsistent with triggered")
        if self.compliance is None and self.normative_references:
            raise ValueError("normative references require a compliance result")
        if self.compliance is not None:
            expected = [
                item.model_dump(mode="json") for item in self.compliance.evidence
            ]
            observed = [
                item.model_dump(mode="json") for item in self.normative_references
            ]
            if observed != expected:
                raise ValueError("normative_references must match compliance.evidence")
        return self


class AnalysisResult(SerializableModel):
    """Complete structured result returned by the public analysis function."""

    run_id: str = Field(min_length=1)
    status: ExecutionStatus
    request: AnalysisRequest
    generated_at: datetime = Field(default_factory=utc_now)
    summary: str
    metrics: TransactionMetrics | None = None
    risk: RiskScore | None = None
    compliance: ComplianceResult | None = None
    charts: list[ChartArtifact] = Field(default_factory=list)
    alert: RiskAlert | None = None
    trace: AnalysisTrace
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


__all__ = [
    "AnalysisRequest",
    "AnalysisResult",
    "AnalysisTrace",
    "BindingStatus",
    "ChartArtifact",
    "ChartType",
    "ComplianceResult",
    "ComplianceStatus",
    "ExecutionStatus",
    "FactorName",
    "RegulatoryEvidence",
    "ReviewStatus",
    "RiskAlert",
    "RiskFactor",
    "RiskFactors",
    "RiskLevel",
    "RiskScore",
    "ScoreThresholds",
    "SerializableModel",
    "TraceEvent",
    "TraceStatus",
    "TransactionMetrics",
    "utc_now",
]
