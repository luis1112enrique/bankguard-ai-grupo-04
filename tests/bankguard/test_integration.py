from __future__ import annotations

import json
import os

import pytest

from agents.bankguard.compliance import build_compliance_agent, invoke_compliance_query
from agents.bankguard.config import load_settings
from agents.bankguard.regulatory_search import create_search_banking_regulations_tool
from agents.bankguard.schemas import ComplianceStatus

LIVE_INTEGRATION = os.getenv("RUN_BANKGUARD_INTEGRATION") == "1"


@pytest.mark.integration
@pytest.mark.skipif(
    not LIVE_INTEGRATION,
    reason="Set RUN_BANKGUARD_INTEGRATION=1 to use local Ollama and Pinecone.",
)
def test_live_compliance_agent_retrieves_the_existing_pinecone_corpus():
    settings = load_settings()
    if not settings.index_name:
        pytest.skip("INDEX_NAME is not configured")

    search_tool = create_search_banking_regulations_tool(settings)
    raw_search = search_tool.invoke(
        {
            "query": (
                "enfoque basado en riesgo PLD FT instituciones bancarias "
                "obligaciones y metodología"
            ),
            "k": 3,
        }
    )
    search = json.loads(raw_search)
    assert search["status"] == "ok"
    assert search["evidence"]
    assert all(
        item["metadata"].get("corpus_id") == settings.corpus_id
        for item in search["evidence"]
    )

    compliance_agent = build_compliance_agent(settings)
    result = invoke_compliance_query(
        compliance_agent,
        (
            "¿Qué fuentes del corpus describen el enfoque basado en riesgo "
            "para PLD/FT en instituciones bancarias?"
        ),
    )

    assert result.status == ComplianceStatus.SUPPORTED
    assert result.evidence
    assert result.retrieval_query and result.retrieval_query.strip()
    assert all(
        item.metadata.get("corpus_id") == settings.corpus_id for item in result.evidence
    )
    assert all(item.source and item.citation for item in result.evidence)
