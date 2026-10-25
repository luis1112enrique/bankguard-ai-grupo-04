from __future__ import annotations

import json

from langchain_core.messages import AIMessage, ToolMessage

from agents.bankguard.compliance import extract_final_text, invoke_compliance_query
from agents.bankguard.schemas import ComplianceStatus


def _evidence() -> dict:
    return {
        "source": "LIC.txt",
        "citation": "LIC.txt, página 42, artículo 115",
        "excerpt": "Fragmento documental recuperado para la prueba.",
        "title": "Ley de Instituciones de Crédito",
        "document_type": "law",
        "page": 42,
        "article": "115",
        "binding_status": "binding",
        "relevance_score": 0.91,
        "metadata": {"corpus_id": "mx-pld-ft", "source": "LIC.txt"},
    }


def _tool_message(
    *,
    status: str = "ok",
    evidence: list[dict] | None = None,
) -> ToolMessage:
    payload = {
        "status": status,
        "query": "obligaciones PLD/FT",
        "evidence": [_evidence()] if evidence is None else evidence,
        "limitations": ["Validar vigencia en fuentes oficiales."],
    }
    return ToolMessage(
        content=json.dumps(payload, ensure_ascii=False),
        tool_call_id="call-compliance-test",
        name="search_banking_regulations",
    )


class StaticComplianceAgent:
    def __init__(self, messages):
        self.messages = messages

    def invoke(self, _payload):
        return {"messages": self.messages}


def test_extract_final_text_never_promotes_tool_or_pending_tool_call():
    pending = AIMessage(
        content="Borrador que todavía solicita una herramienta.",
        tool_calls=[
            {
                "name": "search_banking_regulations",
                "args": {"query": "artículo 115"},
                "id": "call-pending",
                "type": "tool_call",
            }
        ],
    )

    assert extract_final_text({"messages": [_tool_message()]}) == ""
    assert extract_final_text({"messages": [pending, _tool_message()]}) == ""


def test_extract_final_text_accepts_only_completed_textual_assistant_message():
    result = {
        "messages": [
            _tool_message(),
            {"role": "assistant", "content": "Respuesta final verificada."},
        ]
    }

    assert extract_final_text(result) == "Respuesta final verificada."


def test_unavailable_retrieval_discards_hallucinated_model_answer():
    draft = "La ley obliga a reportar cualquier operación superior a 50,000 MXN."
    agent = StaticComplianceAgent(
        [_tool_message(status="unavailable", evidence=[]), AIMessage(content=draft)]
    )

    result = invoke_compliance_query(agent, "¿Qué obligación aplica?")

    assert result.status == ComplianceStatus.ERROR
    assert result.evidence == []
    assert draft not in result.answer
    assert "no estuvo disponible" in result.answer.casefold()


def test_no_evidence_discards_model_answer_and_is_insufficient():
    draft = "Existe un umbral obligatorio no documentado."
    agent = StaticComplianceAgent(
        [
            _tool_message(status="insufficient_evidence", evidence=[]),
            AIMessage(content=draft),
        ]
    )

    result = invoke_compliance_query(agent, "¿Existe un umbral obligatorio?")

    assert result.status == ComplianceStatus.INSUFFICIENT_EVIDENCE
    assert draft not in result.answer
    assert "no se recuperó evidencia" in result.answer.casefold()


def test_tool_only_run_is_not_a_supported_interpretation():
    agent = StaticComplianceAgent([_tool_message()])

    result = invoke_compliance_query(agent, "¿Qué dispone el artículo 115?")

    assert result.status == ComplianceStatus.INSUFFICIENT_EVIDENCE
    assert result.evidence
    assert not result.answer.lstrip().startswith("{")
    assert "no se obtuvo una interpretación" in result.answer.casefold()


def test_answer_without_recovered_source_requires_review_and_is_discarded():
    # "aplica" contiene la secuencia "lic", pero no es una mención a LIC.
    draft = "La regla aplica una obligación general de control."
    agent = StaticComplianceAgent([_tool_message(), AIMessage(content=draft)])

    result = invoke_compliance_query(agent, "¿Qué obligación aplica?")

    assert result.status == ComplianceStatus.REQUIRES_REVIEW
    assert draft not in result.answer
    assert "no pudo validarse" in result.answer.casefold()


def test_unrecovered_article_or_page_requires_review_and_is_discarded():
    draft = (
        "Según LIC.txt, el artículo 999, página 999, obliga a bloquear la cuenta "
        "durante 90 días."
    )
    agent = StaticComplianceAgent([_tool_message(), AIMessage(content=draft)])

    result = invoke_compliance_query(agent, "¿Se debe bloquear la cuenta?")

    assert result.status == ComplianceStatus.REQUIRES_REVIEW
    assert draft not in result.answer
    assert any("no estaban presentes" in caveat for caveat in result.caveats)


def test_answer_with_recovered_source_article_and_page_is_supported():
    answer = (
        "Según LIC.txt, página 42, artículo 115, el fragmento recuperado debe "
        "revisarse junto con su vigencia oficial."
    )
    agent = StaticComplianceAgent([_tool_message(), AIMessage(content=answer)])

    result = invoke_compliance_query(agent, "¿Qué dispone el artículo 115?")

    assert result.status == ComplianceStatus.SUPPORTED
    assert result.answer == answer
    assert len(result.evidence) == 1
