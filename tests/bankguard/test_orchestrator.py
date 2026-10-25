from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import ValidationError

from agents.bankguard.orchestrator import (
    DelegationState,
    compact_conversation_history,
    create_invoke_compliance_agent_tool,
    run_analysis,
)
from agents.bankguard.schemas import (
    AnalysisRequest,
    AnalysisResult,
    ComplianceStatus,
    ExecutionStatus,
    RiskLevel,
)


def _search_payload() -> dict:
    return {
        "status": "ok",
        "query": "obligaciones PLD/FT",
        "evidence": [
            {
                "source": "LIC.txt",
                "citation": "LIC.txt, página 42, artículo 115",
                "excerpt": "Fragmento normativo ficticio usado por la prueba.",
                "document_type": "law",
                "authority": "Cámara de Diputados",
                "page": 42,
                "article": "115",
                "binding_status": "binding",
                "relevance_score": 0.9,
                "metadata": {
                    "corpus_id": "mx-pld-ft",
                    "source": "LIC.txt",
                },
            }
        ],
        "limitations": [
            "La vigencia y consolidación deben validarse en fuentes oficiales."
        ],
    }


class SpyComplianceAgent:
    def __init__(self):
        self.calls = []

    def invoke(self, payload):
        self.calls.append(payload)
        return {
            "messages": [
                ToolMessage(
                    content=json.dumps(_search_payload(), ensure_ascii=False),
                    tool_call_id="call-unit-test",
                    name="search_banking_regulations",
                ),
                AIMessage(
                    content=(
                        "Según LIC.txt, página 42, artículo 115, la conclusión "
                        "requiere validación humana de vigencia."
                    )
                ),
            ]
        }


class FailingComplianceAgent:
    def invoke(self, payload):
        raise ConnectionError("simulated Ollama outage")


class UncitedComplianceAgent:
    def invoke(self, payload):
        return {
            "messages": [
                ToolMessage(
                    content=json.dumps(_search_payload(), ensure_ascii=False),
                    tool_call_id="call-uncited-test",
                    name="search_banking_regulations",
                ),
                AIMessage(content="La obligación aplica de manera general."),
            ]
        }


def test_public_request_and_delegation_share_question_length_limit():
    valid = AnalysisRequest(
        client_id="C003",
        start_date="2026-10-01",
        end_date="2026-10-31",
        compliance_question="x" * 2_000,
    )
    assert len(valid.compliance_question or "") == 2_000

    with pytest.raises(ValidationError):
        AnalysisRequest(
            client_id="C003",
            start_date="2026-10-01",
            end_date="2026-10-31",
            compliance_question="x" * 2_001,
        )
    for too_short in ("x", "xx", "  xx  "):
        with pytest.raises(ValidationError):
            AnalysisRequest(
                client_id="C003",
                start_date="2026-10-01",
                end_date="2026-10-31",
                compliance_question=too_short,
            )


def test_conversation_history_discards_tool_payloads_and_intermediate_calls():
    history = compact_conversation_history(
        [
            {"role": "user", "content": "Analiza C003"},
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "calculate_risk_indicators",
                        "args": {"client_id": "C003"},
                        "id": "call-history-test",
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage(
                content='{"secret_tool_payload": "large evidence"}',
                tool_call_id="call-history-test",
                name="calculate_risk_indicators",
            ),
            AIMessage(content="Resultado final verificado."),
        ],
        max_turns=4,
    )

    assert history == [
        {"role": "user", "content": "Analiza C003"},
        {"role": "assistant", "content": "Resultado final verificado."},
    ]
    assert "secret_tool_payload" not in json.dumps(history)


def test_delegation_tool_really_invokes_specialist_with_one_message():
    spy = SpyComplianceAgent()
    state = DelegationState()
    delegation = create_invoke_compliance_agent_tool(spy, state=state)

    payload = json.loads(
        delegation.invoke(
            {
                "question": "¿Qué obligaciones PLD/FT aplican?",
                "case_context": "Cliente ficticio C003 con patrón atípico.",
            }
        )
    )

    assert len(spy.calls) == 1
    forwarded = spy.calls[0]
    assert list(forwarded) == ["messages"]
    assert len(forwarded["messages"]) == 1
    assert forwarded["messages"][0]["role"] == "user"
    assert "Pregunta normativa" in forwarded["messages"][0]["content"]
    assert "Contexto del caso simulado" in forwarded["messages"][0]["content"]
    assert payload["status"] == ComplianceStatus.SUPPORTED.value
    assert payload["evidence"][0]["article"] == "115"
    assert state.last_compliance is not None


def test_delegation_evidence_is_consumed_once_and_can_be_reset():
    spy = SpyComplianceAgent()
    state = DelegationState()
    delegation = create_invoke_compliance_agent_tool(spy, state=state)

    delegation.invoke({"question": "¿Qué obligaciones PLD/FT aplican?"})

    first = state.consume()
    assert first is not None
    assert state.consume() is None

    delegation.invoke({"question": "¿Qué obligaciones PLD/FT aplican?"})
    assert state.last_compliance is not None
    state.reset()
    assert state.last_compliance is None


def test_run_analysis_needs_no_external_services_when_dependencies_are_injected(
    synthetic_repository,
    isolated_settings,
    monkeypatch,
):
    spy = SpyComplianceAgent()

    def fail_if_external_agent_is_built(*args, **kwargs):
        raise AssertionError("run_analysis tried to construct an external agent")

    monkeypatch.setattr(
        "agents.bankguard.orchestrator.build_compliance_agent",
        fail_if_external_agent_is_built,
    )
    result = run_analysis(
        "C003",
        "2026-10-01",
        "2026-10-31",
        compliance_question="¿Qué obligaciones PLD/FT son relevantes?",
        include_compliance=True,
        include_charts=False,
        generate_alert=False,
        settings=isolated_settings,
        repository=synthetic_repository,
        compliance_agent=spy,
    )

    assert isinstance(result, AnalysisResult)
    assert result.request.client_id == "C003"
    assert result.metrics is not None
    assert result.metrics.transaction_count == 28
    assert result.risk is not None
    assert result.risk.level == RiskLevel.HIGH
    assert result.risk.score is not None and result.risk.score >= 70
    assert result.compliance is not None
    assert result.compliance.status == ComplianceStatus.SUPPORTED
    assert result.charts == []
    assert result.alert is None
    assert len(spy.calls) == 1
    serialized = json.loads(result.to_json())
    assert serialized["risk"]["level"] == "alto"
    assert serialized["trace"]["events"]


def test_run_analysis_degrades_safely_when_compliance_agent_fails(
    synthetic_repository,
    isolated_settings,
):
    result = run_analysis(
        "C003",
        "2026-10-01",
        "2026-10-31",
        include_compliance=True,
        include_charts=False,
        generate_alert=False,
        settings=isolated_settings,
        repository=synthetic_repository,
        compliance_agent=FailingComplianceAgent(),
    )

    assert result.status.value == "partial"
    assert result.risk is not None
    assert result.risk.level == RiskLevel.HIGH
    assert result.compliance is not None
    assert result.compliance.status == ComplianceStatus.ERROR
    assert result.compliance.evidence == []
    assert result.errors == ["La consulta regulatoria no estuvo disponible."]


def test_inconclusive_risk_with_compliance_failure_is_partial(
    synthetic_repository,
    isolated_settings,
):
    result = run_analysis(
        "C006",
        "2026-10-01",
        "2026-10-31",
        include_compliance=True,
        include_charts=False,
        generate_alert=False,
        settings=isolated_settings,
        repository=synthetic_repository,
        compliance_agent=FailingComplianceAgent(),
    )

    assert result.risk is not None
    assert result.risk.level == RiskLevel.INCONCLUSIVE
    assert result.compliance is not None
    assert result.compliance.status == ComplianceStatus.ERROR
    assert result.status == ExecutionStatus.PARTIAL


def test_inconclusive_risk_with_supported_compliance_stays_inconclusive(
    synthetic_repository,
    isolated_settings,
):
    result = run_analysis(
        "C006",
        "2026-10-01",
        "2026-10-31",
        include_compliance=True,
        include_charts=False,
        generate_alert=False,
        settings=isolated_settings,
        repository=synthetic_repository,
        compliance_agent=SpyComplianceAgent(),
    )

    assert result.risk is not None
    assert result.risk.level == RiskLevel.INCONCLUSIVE
    assert result.compliance is not None
    assert result.compliance.status == ComplianceStatus.SUPPORTED
    assert result.status == ExecutionStatus.INCONCLUSIVE


def test_unverified_compliance_answer_makes_analysis_partial(
    synthetic_repository,
    isolated_settings,
):
    result = run_analysis(
        "C003",
        "2026-10-01",
        "2026-10-31",
        include_compliance=True,
        include_charts=False,
        generate_alert=False,
        settings=isolated_settings,
        repository=synthetic_repository,
        compliance_agent=UncitedComplianceAgent(),
    )

    assert result.risk is not None and result.risk.conclusive
    assert result.compliance is not None
    assert result.compliance.status == ComplianceStatus.REQUIRES_REVIEW
    assert result.status == ExecutionStatus.PARTIAL


def test_run_analysis_returns_structured_error_for_unknown_client(
    synthetic_repository,
    isolated_settings,
):
    result = run_analysis(
        "UNKNOWN",
        "2026-10-01",
        "2026-10-31",
        include_compliance=False,
        include_charts=False,
        generate_alert=False,
        settings=isolated_settings,
        repository=synthetic_repository,
    )

    assert result.status == ExecutionStatus.ERROR
    assert result.metrics is None
    assert result.risk is None
    assert result.errors
    assert result.errors == [
        "client_not_found: El cliente solicitado no existe en el dataset."
    ]
    assert result.trace.completed_at is not None
