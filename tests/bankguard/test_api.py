from fastapi.testclient import TestClient

from agents.bankguard import api

client = TestClient(api.app)


def test_health_reports_local_model() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["service"] == "BankGuard AI API"
    assert payload["model"]
    assert payload["max_output_tokens"] == 300


def test_chat_contract(monkeypatch) -> None:
    monkeypatch.setattr(
        api,
        "_run_chat_sync",
        lambda request: api.ChatResponse(
            response=f"Respuesta para: {request.message}",
            sources=["LIC.txt, artículo 115"],
            sql=None,
        ),
    )

    response = client.post(
        "/chat",
        json={
            "message": "  ¿Qué regla aplica?  ",
            "history": [{"role": "user", "content": "Contexto anterior"}],
            "max_output_tokens": 200,
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "response": "Respuesta para: ¿Qué regla aplica?",
        "sources": ["LIC.txt, artículo 115"],
        "sql": None,
    }


def test_chat_rejects_excess_context_and_tokens() -> None:
    too_much_history = [
        {"role": "user", "content": f"mensaje {index}"} for index in range(5)
    ]
    response = client.post(
        "/chat",
        json={
            "message": "pregunta",
            "history": too_much_history,
            "max_output_tokens": 301,
        },
    )

    assert response.status_code == 422


def test_chat_rejects_blank_or_oversized_questions() -> None:
    assert client.post("/chat", json={"message": "   "}).status_code == 422
    assert client.post("/chat", json={"message": "x" * 501}).status_code == 422


def test_chat_rejects_unknown_roles_and_fields() -> None:
    response = client.post(
        "/chat",
        json={
            "message": "pregunta",
            "history": [{"role": "tool", "content": "contenido"}],
            "unexpected": True,
        },
    )

    assert response.status_code == 422


def test_chat_returns_safe_service_error(monkeypatch) -> None:
    def fail(_request):
        raise RuntimeError("secret internal failure")

    monkeypatch.setattr(api, "_run_chat_sync", fail)
    response = client.post("/chat", json={"message": "pregunta válida"})

    assert response.status_code == 503
    assert "secret internal failure" not in response.text


def test_cors_preflight_accepts_local_frontend() -> None:
    response = client.options(
        "/chat",
        headers={
            "Origin": "http://127.0.0.1:8765",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == ("http://127.0.0.1:8765")
