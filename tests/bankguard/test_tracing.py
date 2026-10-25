from __future__ import annotations

import pytest

from agents.bankguard.schemas import TraceStatus
from agents.bankguard.tracing import TraceRecorder, _safe_summary


def test_safe_summary_redacts_mapping_and_authorization_values():
    mapping = _safe_summary(
        {
            "api_key": "abc123",
            "nested": {
                "password": "top-secret",
                "message": "Authorization: Bearer bearer-value",
            },
        }
    )
    text = _safe_summary(
        "api_key='abc123' Authorization: Bearer bearer-value token=last-secret"
    )

    for output in (mapping, text):
        assert output is not None
        assert "abc123" not in output
        assert "top-secret" not in output
        assert "bearer-value" not in output
        assert "last-secret" not in output
        assert "<redacted>" in output


def test_trace_recorder_sanitizes_metadata_and_errors():
    recorder = TraceRecorder("run-redaction-test")
    recorder.record(
        "test",
        "redact",
        TraceStatus.ERROR,
        error="Authorization: Bearer hidden-token",
        metadata={
            "token": "metadata-token",
            "nested": {"api-key": "nested-token", "safe": "retained"},
        },
    )

    event = recorder.snapshot().events[0]
    serialized = event.model_dump_json()
    assert "hidden-token" not in serialized
    assert "metadata-token" not in serialized
    assert "nested-token" not in serialized
    assert event.metadata["token"] == "<redacted>"
    assert event.metadata["nested"]["safe"] == "retained"


@pytest.mark.parametrize(
    "secret_key",
    [
        "PINECONE_API_KEY",
        "OPENAI_API_KEY",
        "access_token",
        "refresh_token",
        "client_secret",
        "X-API-Key",
        "Authorization",
        "cookie",
        "set-cookie",
    ],
)
def test_safe_summary_redacts_prefixed_secret_mapping_keys(secret_key):
    output = _safe_summary(
        {
            secret_key: "credential-that-must-not-leak",
            "safe_field": "safe-value-must-remain",
            "token_count": 3,
        }
    )

    assert output is not None
    assert "credential-that-must-not-leak" not in output
    assert "<redacted>" in output
    assert "safe-value-must-remain" in output
    assert '"token_count": 3' in output


def test_safe_summary_redacts_headers_bearer_tokens_and_cookies():
    output = _safe_summary(
        "\n".join(
            [
                "Authorization: Bearer authorization-secret",
                "Proxy-Authorization: Basic basic-authorization-secret",
                "proxy=Bearer standalone-bearer-secret",
                "X-API-Key: header-api-secret",
                "Cookie: session=cookie-secret; theme=dark",
                "Set-Cookie: refresh=set-cookie-secret; HttpOnly",
                "safe_header: safe-header-value",
            ]
        )
    )

    assert output is not None
    for secret in (
        "authorization-secret",
        "basic-authorization-secret",
        "standalone-bearer-secret",
        "header-api-secret",
        "cookie-secret",
        "set-cookie-secret",
    ):
        assert secret not in output
    assert "safe-header-value" in output
    assert "theme=dark" not in output


def test_safe_summary_redacts_secret_url_query_params_and_preserves_safe_params():
    output = _safe_summary(
        "GET https://api.example.test/search?query=clientes&"
        "PINECONE_API_KEY=query-api-secret&access_token=query-token-secret&"
        "page=2#results"
    )

    assert output is not None
    assert "query-api-secret" not in output
    assert "query-token-secret" not in output
    assert "query=clientes" in output
    assert "page=2#results" in output
    assert output.count("<redacted>") == 2


def test_span_error_records_only_exception_type_not_message_path_or_endpoint():
    recorder = TraceRecorder("run-private-error-test")
    private_path = "/Users/private/customer-data.csv"
    private_endpoint = "https://internal.example.test/accounts?access_token=secret"

    with pytest.raises(RuntimeError):
        with recorder.span("test", "private_failure"):
            raise RuntimeError(f"failed reading {private_path} via {private_endpoint}")

    event = recorder.snapshot().events[0]
    assert event.status == TraceStatus.ERROR
    assert event.error == "RuntimeError"
    serialized = event.model_dump_json()
    assert private_path not in serialized
    assert "internal.example.test" not in serialized
    assert "secret" not in serialized
