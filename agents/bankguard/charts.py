"""Headless, reproducible visualizations for BankGuard analysis results."""

from __future__ import annotations

import hashlib
import json
import uuid
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pandas as pd
from langchain_core.tools import BaseTool, tool
from matplotlib.figure import Figure

from .schemas import ChartArtifact, ChartType
from .tracing import TraceRecorder
from .transaction_tools import BankingDataRepository, ClientPeriodInput


def _artifact_name(client_id: str, kind: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{client_id.lower()}_{kind}_{timestamp}_{uuid.uuid4().hex[:8]}.png"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(64 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _display_path(path: Path, project_root: Path | None) -> str:
    resolved = path.resolve()
    if project_root is not None:
        try:
            return resolved.relative_to(project_root.resolve()).as_posix()
        except ValueError:
            pass
    return resolved.as_posix()


def generate_transaction_charts(
    repository: BankingDataRepository,
    client_id: str,
    start_date: date,
    end_date: date,
    *,
    output_dir: Path,
    project_root: Path | None = None,
) -> list[ChartArtifact]:
    """Create evolution and client-risk-distribution PNGs from real calculations."""
    client = client_id.strip().upper()
    output_dir.mkdir(parents=True, exist_ok=True)
    transactions, period = repository.get_transactions(
        client,
        start_date=start_date,
        end_date=end_date,
    )
    assessment, _ = repository.calculate(
        client,
        start_date=period.start_date,
        end_date=period.end_date,
    )
    currency = assessment.get("metrics", {}).get("currency")
    atypical_ids = {
        item["transaction_id"] for item in assessment["atypical_transactions"]
    }

    evolution_path = output_dir / _artifact_name(client, "evolucion")
    figure = Figure(figsize=(10, 5.5))
    axis = figure.subplots()
    atypical_records: list[dict[str, object]] = []
    if transactions.empty:
        axis.text(
            0.5,
            0.5,
            "Sin transacciones en el periodo",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
        daily_records: list[dict[str, object]] = []
    else:
        daily = transactions.groupby("date", as_index=False)["amount"].sum()
        daily_records = [
            {
                "date": pd.Timestamp(row.date).date().isoformat(),
                "total_amount": round(float(row.amount), 2),
                "currency": currency,
            }
            for row in daily.itertuples(index=False)
        ]
        axis.plot(
            daily["date"],
            daily["amount"],
            marker="o",
            linewidth=1.8,
            color="#1f4e79",
            label="Monto diario",
        )
        unusual = transactions[
            transactions["transaction_id"].astype(str).isin(atypical_ids)
        ]
        atypical_records = [
            {
                "transaction_id": str(row.transaction_id),
                "date": pd.Timestamp(row.date).date().isoformat(),
                "amount": round(float(row.amount), 2),
                "currency": str(row.currency),
            }
            for row in unusual.itertuples(index=False)
        ]
        if not unusual.empty:
            axis.scatter(
                unusual["date"],
                unusual["amount"],
                color="#c0392b",
                marker="x",
                s=55,
                linewidths=1.8,
                label="Operación atípica",
                zorder=3,
            )
        axis.legend(loc="upper left")
    axis.set_title(f"Evolución de transacciones de {client}")
    axis.set_xlabel("Fecha")
    axis.set_ylabel(f"Monto ({currency})" if currency else "Monto")
    axis.grid(axis="y", alpha=0.25)
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(evolution_path, dpi=150)

    order = ["bajo", "medio", "alto", "no_concluyente"]
    distribution: Counter[str] = Counter()
    per_client: list[dict[str, object]] = []
    for candidate in repository.client_ids:
        candidate_result, _ = repository.calculate(
            candidate,
            start_date=period.start_date,
            end_date=period.end_date,
        )
        level = str(candidate_result["risk_level"])
        distribution[level] += 1
        per_client.append(
            {
                "client_id": candidate,
                "risk_level": level,
                "risk_score": candidate_result["risk_score"],
                "assessment_status": candidate_result["assessment_status"],
            }
        )

    distribution_path = output_dir / _artifact_name(client, "distribucion_riesgo")
    distribution_figure = Figure(figsize=(8, 5))
    distribution_axis = distribution_figure.subplots()
    colors = ["#2e8b57", "#d4a017", "#c0392b", "#7f8c8d"]
    labels = ["Bajo", "Medio", "Alto", "No concluyente"]
    values = [distribution[key] for key in order]
    bars = distribution_axis.bar(labels, values, color=colors)
    distribution_axis.bar_label(bars, padding=3)
    distribution_axis.set_title("Distribución de riesgo por cliente")
    distribution_axis.set_xlabel("Nivel de riesgo simulado")
    distribution_axis.set_ylabel("Número de clientes")
    distribution_axis.set_ylim(0, max(values + [1]) + 1)
    distribution_axis.grid(axis="y", alpha=0.2)
    distribution_figure.tight_layout()
    distribution_figure.savefig(distribution_path, dpi=150)

    common_period = {
        "start_date": period.start_date.isoformat(),
        "end_date": period.end_date.isoformat(),
    }
    return [
        ChartArtifact(
            chart_type=ChartType.TRANSACTION_EVOLUTION,
            relative_path=_display_path(evolution_path, project_root),
            title=f"Evolución de transacciones de {client}",
            description=(
                "Totales diarios; las marcas rojas corresponden a operaciones "
                "detectadas por reglas deterministas del demo."
            ),
            sha256=_sha256(evolution_path),
            metadata={
                "client_id": client,
                "period": common_period,
                "currency": currency,
                "unit_of_analysis": "daily_amount_and_individual_atypical_transactions",
                "daily_totals": daily_records,
                "atypical_transaction_ids": sorted(atypical_ids),
                "atypical_transactions": atypical_records,
            },
        ),
        ChartArtifact(
            chart_type=ChartType.RISK_DISTRIBUTION,
            relative_path=_display_path(distribution_path, project_root),
            title="Distribución de niveles de riesgo por cliente",
            description=(
                "Conteo de clientes por nivel para el mismo periodo; los niveles "
                "son criterios simulados y no umbrales regulatorios."
            ),
            sha256=_sha256(distribution_path),
            metadata={
                "period": common_period,
                "unit_of_analysis": "clients",
                "counts": dict(distribution),
                "clients": per_client,
            },
        ),
    ]


def create_generate_transaction_chart_tool(
    repository: BankingDataRepository,
    *,
    output_dir: Path,
    project_root: Path | None = None,
    recorder: TraceRecorder | None = None,
) -> BaseTool:
    """Create the LangChain chart tool backed by validated repository data."""
    trace = recorder or TraceRecorder()

    @tool("generate_transaction_chart", args_schema=ClientPeriodInput)
    def generate_transaction_chart(
        client_id: str,
        start_date: date | None = None,
        end_date: date | None = None,
        last_n_days: int | None = None,
    ) -> str:
        """Genera evolución de montos y distribución de riesgo en archivos PNG."""
        period = repository.resolve_period(
            start_date=start_date,
            end_date=end_date,
            last_n_days=last_n_days,
        )
        summary = (
            f"client_id={client_id}; start={period.start_date}; end={period.end_date}"
        )
        with trace.span(
            "charts", "generate_transaction_chart", input_summary=summary
        ) as span:
            artifacts = generate_transaction_charts(
                repository,
                client_id,
                period.start_date,
                period.end_date,
                output_dir=output_dir,
                project_root=project_root,
            )
            span.output_summary = f"chart_count={len(artifacts)}"
            return json.dumps(
                {"charts": [item.to_dict() for item in artifacts]},
                ensure_ascii=False,
            )

    return generate_transaction_chart


__all__ = [
    "create_generate_transaction_chart_tool",
    "generate_transaction_charts",
]
