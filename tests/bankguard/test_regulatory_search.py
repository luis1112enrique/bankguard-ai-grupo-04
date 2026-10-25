from __future__ import annotations

import json

from langchain_core.documents import Document

from agents.bankguard.regulatory_search import create_search_banking_regulations_tool

SETTINGS = {
    "index_name": "fake-index",
    "corpus_id": "mx-pld-ft",
    "pinecone_namespace": "mx-pld-ft",
    "regulatory_retrieval_k": 5,
}


class FakeBackend:
    def __init__(self):
        self.calls = []

    def search(self, query, *, k, metadata_filter):
        self.calls.append((query, k, metadata_filter))
        return [
            (
                Document(
                    page_content="Las instituciones deben observar la disposición.",
                    metadata={
                        "corpus_id": "mx-pld-ft",
                        "source": "Disposiciones_Articulo_115_LIC.txt",
                        "title": "Disposiciones del artículo 115",
                        "document_type": "regulation",
                        "authority": "SHCP",
                        "jurisdiction": "México",
                        "page": 12,
                        "provision": "4ª",
                        "citation": (
                            "Disposiciones_Articulo_115_LIC.txt, página 12, "
                            "disposición 4ª"
                        ),
                        "chunk_sha256": "law-chunk",
                    },
                ),
                0.91,
            ),
            (
                Document(
                    page_content="La guía describe una metodología de riesgo.",
                    metadata={
                        "corpus_id": "mx-pld-ft",
                        "source": "CNBV_Guia_Metodologia_Evaluacion_Riesgos_PLD_FT_2019.txt",
                        "title": "Guía metodológica CNBV",
                        "document_type": "methodology_guide",
                        "authority": "CNBV",
                        "page": 8,
                        "citation": "Guía CNBV, página 8",
                        "chunk_sha256": "guide-chunk",
                    },
                ),
                0.82,
            ),
            Document(
                page_content="Este fragmento debe descartarse.",
                metadata={
                    "corpus_id": "otro-corpus",
                    "source": "otro.txt",
                    "document_type": "law",
                },
            ),
        ]


class FailingBackend:
    def search(self, query, *, k, metadata_filter):
        raise RuntimeError("super-secret-endpoint")


def test_fake_retriever_preserves_metadata_and_enforces_corpus_filter():
    backend = FakeBackend()
    search = create_search_banking_regulations_tool(SETTINGS, backend=backend)

    payload = json.loads(
        search.invoke(
            {
                "query": "  obligaciones   de monitoreo transaccional  ",
                "k": 8,
            }
        )
    )

    assert backend.calls == [
        (
            "obligaciones de monitoreo transaccional",
            5,
            {"corpus_id": {"$eq": "mx-pld-ft"}},
        )
    ]
    assert payload["status"] == "ok"
    assert payload["namespace"] == "mx-pld-ft"
    assert len(payload["evidence"]) == 2
    law, guidance = payload["evidence"]
    assert law["binding_status"] == "binding"
    assert law["metadata"]["authority"] == "SHCP"
    assert law["provision"] == "4ª"
    assert law["relevance_score"] == 0.91
    assert guidance["binding_status"] == "guidance"
    assert any("2009" in limitation for limitation in payload["limitations"])
    assert any(
        "Se descartaron 1" in limitation for limitation in payload["limitations"]
    )


def test_retriever_failure_is_safe_and_does_not_leak_exception_message():
    search = create_search_banking_regulations_tool(
        SETTINGS,
        backend=FailingBackend(),
    )

    raw = search.invoke({"query": "obligaciones PLD FT"})
    payload = json.loads(raw)

    assert payload["status"] == "unavailable"
    assert payload["evidence"] == []
    assert payload["error_type"] == "RuntimeError"
    assert "super-secret-endpoint" not in raw
