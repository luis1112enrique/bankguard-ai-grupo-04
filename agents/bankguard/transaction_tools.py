"""Validated transaction access and deterministic LangChain tools."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd
from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field, model_validator

from .config import BankGuardSettings, RiskEngineConfig, load_risk_config
from .risk_engine import RiskEngine
from .tracing import TraceRecorder


class ClientPeriodInput(BaseModel):
    """Safe client/period arguments exposed to the language model."""

    client_id: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9_-]+$",
        description="Identificador ficticio, por ejemplo C003.",
    )
    start_date: date | None = Field(
        default=None,
        description="Fecha inicial inclusiva en formato YYYY-MM-DD.",
    )
    end_date: date | None = Field(
        default=None,
        description="Fecha final inclusiva en formato YYYY-MM-DD.",
    )
    last_n_days: int | None = Field(
        default=None,
        ge=1,
        le=366,
        description=(
            "Periodo relativo inclusivo. Sin fecha de corte usa la fecha máxima "
            "del dataset para que el demo sea reproducible."
        ),
    )

    @model_validator(mode="after")
    def validate_period(self) -> "ClientPeriodInput":
        if self.start_date and self.end_date and self.end_date < self.start_date:
            raise ValueError("end_date must be on or after start_date")
        if self.last_n_days is not None and self.start_date is not None:
            raise ValueError("Use either start_date or last_n_days, not both")
        self.client_id = self.client_id.strip().upper()
        return self


@dataclass(frozen=True, slots=True)
class ResolvedPeriod:
    start_date: date
    end_date: date
    reference_date: date
    policy: str


class BankingDataRepository:
    """Own the validated synthetic dataset used by every transaction tool."""

    def __init__(
        self,
        transaction_path: str | Any,
        profile_path: str | Any,
        *,
        risk_config: RiskEngineConfig | None = None,
    ) -> None:
        self.transaction_path = transaction_path
        self.profile_path = profile_path
        self.risk_config = risk_config or RiskEngineConfig()
        self.engine = RiskEngine.from_csv(
            transaction_path,
            profile_path,
            config=self.risk_config,
        )

    @classmethod
    def from_settings(cls, settings: BankGuardSettings) -> "BankingDataRepository":
        return cls(
            settings.transactions_csv,
            settings.profiles_csv,
            risk_config=load_risk_config(settings.risk_config_path),
        )

    @property
    def reference_date(self) -> date:
        if self.engine.transactions.empty:
            raise ValueError("The transaction dataset is empty")
        return pd.Timestamp(self.engine.transactions["date"].max()).date()

    @property
    def client_ids(self) -> list[str]:
        return sorted(self.engine.profiles["client_id"].astype(str).tolist())

    def resolve_period(
        self,
        *,
        start_date: date | None = None,
        end_date: date | None = None,
        last_n_days: int | None = None,
    ) -> ResolvedPeriod:
        reference = self.reference_date
        if last_n_days is not None:
            resolved_end = end_date or reference
            resolved_start = resolved_end - pd.Timedelta(days=last_n_days - 1)
            policy = (
                "periodo_relativo_con_fecha_final_explicita"
                if end_date
                else "periodo_relativo_a_fecha_maxima_del_dataset"
            )
        elif start_date is None and end_date is None:
            resolved_end = reference
            resolved_start = resolved_end.replace(day=1)
            policy = "mes_de_la_fecha_maxima_del_dataset"
        elif start_date is None:
            resolved_end = end_date
            resolved_start = resolved_end.replace(day=1)
            policy = "mes_de_fecha_final_explicita"
        else:
            resolved_start = start_date
            resolved_end = end_date or reference
            policy = (
                "periodo_explicito"
                if end_date
                else "inicio_explicito_hasta_fecha_maxima_del_dataset"
            )

        start_value = pd.Timestamp(resolved_start).date()
        end_value = pd.Timestamp(resolved_end).date()
        if end_value < start_value:
            raise ValueError("end_date cannot be earlier than start_date")
        days = (end_value - start_value).days + 1
        if days > self.risk_config.maximum_analysis_days:
            raise ValueError(
                f"Analysis period cannot exceed {self.risk_config.maximum_analysis_days} days"
            )
        return ResolvedPeriod(start_value, end_value, reference, policy)

    def get_transactions(
        self,
        client_id: str,
        *,
        start_date: date | None = None,
        end_date: date | None = None,
        last_n_days: int | None = None,
    ) -> tuple[pd.DataFrame, ResolvedPeriod]:
        period = self.resolve_period(
            start_date=start_date,
            end_date=end_date,
            last_n_days=last_n_days,
        )
        frame = self.engine.get_client_transactions(
            client_id.strip().upper(),
            period.start_date,
            period.end_date,
        )
        return frame, period

    def calculate(
        self,
        client_id: str,
        *,
        start_date: date | None = None,
        end_date: date | None = None,
        last_n_days: int | None = None,
    ) -> tuple[dict[str, Any], ResolvedPeriod]:
        period = self.resolve_period(
            start_date=start_date,
            end_date=end_date,
            last_n_days=last_n_days,
        )
        result = self.engine.calculate_risk_indicators(
            client_id.strip().upper(),
            period.start_date,
            period.end_date,
        )
        result["period"]["reference_date"] = period.reference_date.isoformat()
        result["period"]["resolution_policy"] = period.policy
        return result, period


def frame_to_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert validated transaction rows into JSON-safe records."""
    records: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        row["date"] = pd.Timestamp(row["date"]).date().isoformat()
        row["amount"] = round(float(row["amount"]), 2)
        records.append(row)
    return records


def verified_response_facts(result: dict[str, Any]) -> list[str]:
    """Build deterministic Spanish facts the conversational agent may quote."""

    def number(value: Any, decimals: int = 6) -> str:
        return "no_disponible" if value is None else f"{float(value):.{decimals}f}"

    period = result["period"]
    metrics = result["metrics"]
    historical = result["historical_metrics"]
    score = result.get("risk_score")
    score_text = number(score)
    currency = metrics.get("currency") or "moneda_no_disponible"
    history_start = historical.get("window_start") or "no_disponible"
    history_end = historical.get("window_end") or "no_disponible"
    facts = [
        (
            f"Periodo analizado: {period['start_date']} a {period['end_date']} "
            f"({metrics['period_days']} días)."
        ),
        (
            f"Ventana histórica: {history_start} a "
            f"{history_end} "
            f"({historical.get('coverage_days', 0)} días de cobertura)."
        ),
        (
            f"Operaciones: {metrics['transaction_count']}; monto total: "
            f"{number(metrics['total_amount'], 2)} {currency}; promedio: "
            f"{number(metrics['average_amount'], 2)} {currency}; máximo: "
            f"{number(metrics['maximum_amount'], 2)} {currency}; desviación "
            f"estándar: {number(metrics['standard_deviation'], 2)} {currency}."
        ),
        (
            f"Frecuencia diaria: {number(metrics['daily_frequency'])}; "
            f"frecuencia semanal: {number(metrics['weekly_frequency'])}."
        ),
        (
            f"Variación de monto ajustada por días frente al historial: "
            f"{number(metrics['amount_variation_pct'])}%; variación de "
            f"frecuencia ajustada por días: "
            f"{number(metrics['frequency_variation_pct'])}%."
        ),
        (
            f"Operaciones atípicas: {metrics['atypical_transaction_count']} "
            f"({number(metrics['atypical_transaction_rate_pct'])}%)."
        ),
        (
            f"Evaluación: {result['assessment_status']}; score simulado: "
            f"{score_text}; nivel: {result['risk_level']}."
        ),
    ]
    for name, factor in result.get("risk_factors", {}).items():
        normalized = factor.get("normalized_score")
        contribution = factor.get("contribution")
        normalized_text = number(normalized)
        contribution_text = number(contribution)
        facts.append(
            f"Factor {name}: normalized_score={normalized_text}/100; "
            f"contribution={contribution_text} puntos."
        )
    return facts


def create_transaction_tools(
    repository: BankingDataRepository,
    *,
    recorder: TraceRecorder | None = None,
) -> tuple[BaseTool, BaseTool]:
    """Create the two explicit transaction tools used by the orchestrator."""
    trace = recorder or TraceRecorder()

    @tool("get_client_transactions", args_schema=ClientPeriodInput)
    def get_client_transactions(
        client_id: str,
        start_date: date | None = None,
        end_date: date | None = None,
        last_n_days: int | None = None,
    ) -> str:
        """Recupera transacciones ficticias validadas de un cliente y periodo.

        Si no se indica periodo, usa el mes de la fecha máxima del dataset. Para
        «últimos N días», usa ``last_n_days``; sin corte explícito la referencia
        también es la fecha máxima del dataset, no la fecha actual del sistema.
        """
        summary = f"client_id={client_id}; start={start_date}; end={end_date}; last_n={last_n_days}"
        with trace.span(
            "transaction_tools", "get_client_transactions", input_summary=summary
        ) as span:
            frame, period = repository.get_transactions(
                client_id,
                start_date=start_date,
                end_date=end_date,
                last_n_days=last_n_days,
            )
            payload = {
                "client_id": client_id.strip().upper(),
                "period": {
                    "start_date": period.start_date.isoformat(),
                    "end_date": period.end_date.isoformat(),
                    "reference_date": period.reference_date.isoformat(),
                    "resolution_policy": period.policy,
                },
                "transaction_count": len(frame),
                "transactions": frame_to_records(frame),
                "data_notice": "Datos enteramente ficticios para demostración.",
            }
            span.output_summary = f"transaction_count={len(frame)}"
            return json.dumps(payload, ensure_ascii=False)

    @tool("calculate_risk_indicators", args_schema=ClientPeriodInput)
    def calculate_risk_indicators(
        client_id: str,
        start_date: date | None = None,
        end_date: date | None = None,
        last_n_days: int | None = None,
    ) -> str:
        """Calcula métricas y score deterministas; el LLM no hace las cuentas."""
        summary = f"client_id={client_id}; start={start_date}; end={end_date}; last_n={last_n_days}"
        with trace.span(
            "risk_engine", "calculate_risk_indicators", input_summary=summary
        ) as span:
            result, _ = repository.calculate(
                client_id,
                start_date=start_date,
                end_date=end_date,
                last_n_days=last_n_days,
            )
            span.output_summary = (
                f"status={result['assessment_status']}; score={result['risk_score']}; "
                f"level={result['risk_level']}"
            )
            payload = {
                **result,
                "verified_response_facts": verified_response_facts(result),
                "response_constraints": [
                    (
                        "Usa únicamente verified_response_facts para narrar cifras; "
                        "no calcules porcentajes adicionales."
                    ),
                    (
                        "No compares totales brutos de periodos con distinta duración; "
                        "usa las variaciones ajustadas ya calculadas."
                    ),
                    (
                        "normalized_score es intensidad sobre 100 y contribution son "
                        "puntos ponderados; no intercambies las etiquetas."
                    ),
                ],
            }
            return json.dumps(payload, ensure_ascii=False)

    return get_client_transactions, calculate_risk_indicators


__all__ = [
    "BankingDataRepository",
    "ClientPeriodInput",
    "ResolvedPeriod",
    "create_transaction_tools",
    "frame_to_records",
    "verified_response_facts",
]
