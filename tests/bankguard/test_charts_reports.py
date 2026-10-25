from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from agents.bankguard.charts import generate_transaction_charts
from agents.bankguard.reports import (
    build_risk_alert,
    create_generate_risk_alert_tool,
)
from agents.bankguard.schemas import (
    BindingStatus,
    ComplianceResult,
    ComplianceStatus,
    RegulatoryEvidence,
    ReviewStatus,
    RiskAlert,
)

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def test_charts_are_valid_pngs_and_never_overwrite(synthetic_repository, tmp_path):
    output_dir = tmp_path / "charts"
    first = generate_transaction_charts(
        synthetic_repository,
        "C003",
        start_date=synthetic_repository.resolve_period().start_date,
        end_date=synthetic_repository.resolve_period().end_date,
        output_dir=output_dir,
        project_root=tmp_path,
    )
    second = generate_transaction_charts(
        synthetic_repository,
        "C003",
        start_date=synthetic_repository.resolve_period().start_date,
        end_date=synthetic_repository.resolve_period().end_date,
        output_dir=output_dir,
        project_root=tmp_path,
    )

    assert len(first) == len(second) == 2
    first_paths = {artifact.relative_path for artifact in first}
    second_paths = {artifact.relative_path for artifact in second}
    assert first_paths.isdisjoint(second_paths)
    for artifact in [*first, *second]:
        path = tmp_path / artifact.relative_path
        assert path.is_file()
        assert path.suffix == ".png"
        assert path.read_bytes().startswith(PNG_SIGNATURE)
        assert len(artifact.sha256 or "") == 64

    evolution = first[0]
    assert evolution.metadata["currency"] == "MXN"
    assert all(
        set(record) == {"date", "total_amount", "currency"}
        for record in evolution.metadata["daily_totals"]
    )
    assert all(
        record["currency"] == "MXN" for record in evolution.metadata["daily_totals"]
    )
    atypical_records = evolution.metadata["atypical_transactions"]
    assert atypical_records
    assert {record["transaction_id"] for record in atypical_records} == set(
        evolution.metadata["atypical_transaction_ids"]
    )
    assert all(
        set(record) == {"transaction_id", "date", "amount", "currency"}
        for record in atypical_records
    )
    assert all(record["currency"] == "MXN" for record in atypical_records)


def test_alert_persists_auditable_json_file(synthetic_repository, tmp_path):
    assessment, _ = synthetic_repository.calculate(
        "C003",
        start_date=synthetic_repository.resolve_period().start_date,
        end_date=synthetic_repository.resolve_period().end_date,
    )
    evidence = RegulatoryEvidence(
        source="LIC.txt",
        citation="LIC.txt, página 42, artículo 115",
        excerpt="Fragmento normativo de prueba.",
        document_type="law",
        authority="Cámara de Diputados",
        page=42,
        article="115",
        binding_status=BindingStatus.BINDING,
    )
    compliance = ComplianceResult(
        question="¿Qué revisión corresponde?",
        status=ComplianceStatus.SUPPORTED,
        answer="La evidencia debe someterse a revisión humana.",
        evidence=[evidence],
        binding_sources_used=1,
        human_review_required=True,
    )

    alert = build_risk_alert(
        assessment,
        run_id="run-unit-test",
        output_dir=tmp_path / "alerts",
        project_root=tmp_path,
        compliance=compliance,
    )

    assert alert.triggered is True
    assert alert.level.value == "alto"
    assert alert.score == assessment["risk_score"]
    assert alert.artifact_relative_path is not None
    artifact_path = tmp_path / alert.artifact_relative_path
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert payload["alert_id"] == alert.alert_id
    assert payload["run_id"] == "run-unit-test"
    assert payload["normative_references"][0]["citation"].endswith("artículo 115")
    assert payload["indicators"]["risk_level"] == "alto"
    assert "no constituye una acusación" in payload["disclaimer"]
    assert Path(alert.artifact_relative_path).suffix == ".json"


def test_low_result_is_not_presented_or_persisted_as_review_alert(
    synthetic_repository,
    tmp_path,
):
    period = synthetic_repository.resolve_period()
    assessment, _ = synthetic_repository.calculate(
        "C001",
        start_date=period.start_date,
        end_date=period.end_date,
    )

    alert = build_risk_alert(
        assessment,
        run_id="run-low-risk",
        output_dir=tmp_path / "alerts",
        project_root=tmp_path,
    )

    assert alert.level.value == "bajo"
    assert alert.triggered is False
    assert alert.review_status == ReviewStatus.NOT_TRIGGERED
    assert "sin alerta" in alert.title.casefold()
    assert "no se activó un expediente" in alert.message.casefold()
    assert alert.artifact_relative_path is None
    assert not (tmp_path / "alerts").exists()
    assert alert.recommended_actions == [
        "Continuar el monitoreo ordinario conforme a los controles internos."
    ]


def test_risk_alert_rejects_inconsistent_trigger_and_evidence(
    synthetic_repository,
    tmp_path,
):
    period = synthetic_repository.resolve_period()
    assessment, _ = synthetic_repository.calculate(
        "C003",
        start_date=period.start_date,
        end_date=period.end_date,
    )
    evidence = RegulatoryEvidence(
        source="LIC.txt",
        citation="LIC.txt, página 42, artículo 115",
        excerpt="Fragmento normativo de prueba.",
    )
    compliance = ComplianceResult(
        status=ComplianceStatus.SUPPORTED,
        answer="Conclusión documentada.",
        evidence=[evidence],
    )
    alert = build_risk_alert(
        assessment,
        run_id="run-contract",
        output_dir=tmp_path / "alerts",
        project_root=tmp_path,
        compliance=compliance,
    )

    invalid_trigger = {**alert.to_dict(), "triggered": False}
    with pytest.raises(ValidationError, match="triggered is inconsistent"):
        RiskAlert.model_validate(invalid_trigger)

    invalid_references = {**alert.to_dict(), "normative_references": []}
    with pytest.raises(ValidationError, match="must match compliance.evidence"):
        RiskAlert.model_validate(invalid_references)

    with pytest.raises(ValidationError):
        alert.triggered = False
    assert alert.triggered is True

    with pytest.raises(ValidationError, match="requires retrieved evidence"):
        ComplianceResult(
            status=ComplianceStatus.SUPPORTED,
            answer="Conclusión sin respaldo.",
        )


def test_batch_alerts_never_reuse_one_clients_compliance(
    synthetic_repository,
    tmp_path,
):
    evidence = RegulatoryEvidence(
        source="LIC.txt",
        citation="LIC.txt, página 42, artículo 115",
        excerpt="Fragmento normativo de prueba.",
    )
    compliance = ComplianceResult(
        status=ComplianceStatus.SUPPORTED,
        answer="Conclusión documentada para un caso individual.",
        evidence=[evidence],
    )
    calls = 0

    def supply_compliance():
        nonlocal calls
        calls += 1
        return compliance

    tool = create_generate_risk_alert_tool(
        synthetic_repository,
        run_id="run-batch",
        output_dir=tmp_path / "alerts",
        project_root=tmp_path,
        compliance_supplier=supply_compliance,
    )
    payload = json.loads(
        tool.invoke(
            {
                "client_id": "ALL",
                "start_date": "2026-10-01",
                "end_date": "2026-10-31",
            }
        )
    )

    assert calls == 0
    assert payload["compliance_attached"] is False
    assert "cada cliente requiere" in payload["compliance_note"]
    assert len(payload["alerts"]) == len(synthetic_repository.client_ids)
    assert all(alert["compliance"] is None for alert in payload["alerts"])
    assert all(not alert["normative_references"] for alert in payload["alerts"])

    class SingleClientView:
        client_ids = ["C003"]

        def resolve_period(self, **kwargs):
            return synthetic_repository.resolve_period(**kwargs)

        def calculate(self, client_id, **kwargs):
            assert client_id == "C003"
            return synthetic_repository.calculate(client_id, **kwargs)

    single_client_tool = create_generate_risk_alert_tool(
        SingleClientView(),
        run_id="run-batch-single-client",
        output_dir=tmp_path / "single-client-alerts",
        project_root=tmp_path,
        compliance_supplier=supply_compliance,
    )
    single_payload = json.loads(
        single_client_tool.invoke(
            {
                "client_id": "ALL",
                "start_date": "2026-10-01",
                "end_date": "2026-10-31",
            }
        )
    )

    assert calls == 0
    assert single_payload["compliance_attached"] is False
    assert single_payload["alerts"][0]["compliance"] is None
