"""Structured JSON review files for deterministic BankGuard alerts."""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool, tool

from .schemas import ComplianceResult, ReviewStatus, RiskAlert, RiskLevel
from .tracing import TraceRecorder
from .transaction_tools import BankingDataRepository, ClientPeriodInput


def _display_path(path: Path, project_root: Path | None) -> str:
    resolved = path.resolve()
    if project_root is not None:
        try:
            return resolved.relative_to(project_root.resolve()).as_posix()
        except ValueError:
            pass
    return resolved.as_posix()


def _factor_explanations(assessment: dict[str, Any]) -> list[str]:
    ranked: list[tuple[float, str]] = []
    for name, detail in assessment.get("risk_factors", {}).items():
        value = detail.get("normalized_score")
        if value is None:
            continue
        explanation = str(detail.get("explanation") or name)
        ranked.append((float(value), f"{name}: {float(value):.2f}/100. {explanation}"))
    return [text for _, text in sorted(ranked, reverse=True)]


def build_risk_alert(
    assessment: dict[str, Any],
    *,
    run_id: str,
    output_dir: Path,
    project_root: Path | None = None,
    compliance: ComplianceResult | None = None,
) -> RiskAlert:
    """Create and, when review is warranted, persist one auditable alert."""
    client_id = str(assessment["client_id"])
    period = assessment["period"]
    level = RiskLevel(str(assessment["risk_level"]))
    triggered = level in {
        RiskLevel.MEDIUM,
        RiskLevel.HIGH,
        RiskLevel.INCONCLUSIVE,
    }
    now = datetime.now(timezone.utc)
    alert_id = f"alert-{uuid.uuid4().hex}"
    artifact_path: Path | None = None
    artifact_display: str | None = None
    if triggered:
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = (
            f"{client_id.lower()}_{now.strftime('%Y%m%dT%H%M%S%fZ')}_"
            f"{uuid.uuid4().hex[:8]}.json"
        )
        artifact_path = output_dir / filename
        artifact_display = _display_path(artifact_path, project_root)

    score = assessment.get("risk_score")
    if level == RiskLevel.INCONCLUSIVE:
        message = (
            "La información disponible no permite una clasificación concluyente; "
            "el expediente requiere revisión humana."
        )
    elif not triggered:
        message = (
            f"El score simulado fue {float(score):.2f}/100 ({level.value}). "
            "No se activó un expediente de revisión; el resultado conserva el "
            "monitoreo ordinario y no constituye evidencia de delito."
        )
    else:
        message = (
            f"El score simulado fue {float(score):.2f}/100 ({level.value}). "
            "Es una señal analítica para priorizar revisión, no evidencia de delito."
        )

    title = (
        f"Expediente analítico de revisión para {client_id}"
        if triggered
        else f"Evaluación analítica sin alerta para {client_id}"
    )
    recommended_actions = (
        [
            "Validar la calidad y suficiencia del historial transaccional.",
            "Revisar las operaciones atípicas y su contexto con personal autorizado.",
            "Confirmar obligaciones aplicables en fuentes oficiales vigentes antes de actuar.",
        ]
        if triggered
        else ["Continuar el monitoreo ordinario conforme a los controles internos."]
    )

    metrics = dict(assessment.get("metrics", {}))
    alert = RiskAlert(
        alert_id=alert_id,
        run_id=run_id,
        client_id=client_id,
        created_at=now,
        period_start=date.fromisoformat(period["start_date"]),
        period_end=date.fromisoformat(period["end_date"]),
        triggered=triggered,
        review_status=(
            ReviewStatus.PENDING if triggered else ReviewStatus.NOT_TRIGGERED
        ),
        score=score,
        level=level,
        title=title,
        message=message,
        transaction_summary={
            "transaction_count": metrics.get("transaction_count", 0),
            "currency": metrics.get("currency"),
            "total_amount": metrics.get("total_amount", 0),
            "average_amount": metrics.get("average_amount"),
            "maximum_amount": metrics.get("maximum_amount"),
            "atypical_transaction_count": metrics.get("atypical_transaction_count", 0),
        },
        indicators=assessment,
        contributing_factors=_factor_explanations(assessment),
        atypical_operations=list(assessment.get("atypical_transactions", [])),
        compliance=compliance,
        normative_references=list(compliance.evidence) if compliance else [],
        recommended_actions=recommended_actions,
        artifact_relative_path=artifact_display,
    )

    if artifact_path is not None:
        temporary = artifact_path.with_suffix(".tmp")
        temporary.write_text(alert.to_json(indent=2), encoding="utf-8")
        temporary.replace(artifact_path)
    return alert


def create_generate_risk_alert_tool(
    repository: BankingDataRepository,
    *,
    run_id: str,
    output_dir: Path,
    project_root: Path | None = None,
    compliance_supplier: Callable[[], ComplianceResult | None] | None = None,
    recorder: TraceRecorder | None = None,
) -> BaseTool:
    """Create the alert tool; it can consume the latest delegated evidence."""
    trace = recorder or TraceRecorder(run_id)

    @tool("generate_risk_alert", args_schema=ClientPeriodInput)
    def generate_risk_alert(
        client_id: str,
        start_date: date | None = None,
        end_date: date | None = None,
        last_n_days: int | None = None,
    ) -> str:
        """Genera expedientes JSON; usa client_id='ALL' para todos los clientes."""
        period = repository.resolve_period(
            start_date=start_date,
            end_date=end_date,
            last_n_days=last_n_days,
        )
        batch_requested = client_id.strip().upper() == "ALL"
        clients = (
            repository.client_ids if batch_requested else [client_id.strip().upper()]
        )
        summary = (
            f"client_id={client_id}; start={period.start_date}; end={period.end_date}"
        )
        with trace.span(
            "reports", "generate_risk_alert", input_summary=summary
        ) as span:
            compliance = (
                compliance_supplier()
                if compliance_supplier is not None and not batch_requested
                else None
            )
            alerts = []
            for client in clients:
                assessment, _ = repository.calculate(
                    client,
                    start_date=period.start_date,
                    end_date=period.end_date,
                )
                alerts.append(
                    build_risk_alert(
                        assessment,
                        run_id=run_id,
                        output_dir=output_dir,
                        project_root=project_root,
                        compliance=compliance,
                    )
                )
            triggered = [item for item in alerts if item.triggered]
            span.output_summary = f"evaluated={len(alerts)}; triggered={len(triggered)}"
            return json.dumps(
                {
                    "alerts": [item.to_dict() for item in alerts],
                    "triggered_count": len(triggered),
                    "compliance_attached": compliance is not None,
                    "compliance_note": (
                        None
                        if not batch_requested
                        else (
                            "La evidencia de un caso no se adjunta a alertas batch; "
                            "cada cliente requiere una delegación propia."
                        )
                    ),
                },
                ensure_ascii=False,
            )

    return generate_risk_alert


__all__ = ["build_risk_alert", "create_generate_risk_alert_tool"]
