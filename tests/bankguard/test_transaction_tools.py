from __future__ import annotations

import json
from datetime import date

import pytest
from pydantic import ValidationError

from agents.bankguard.transaction_tools import (
    ClientPeriodInput,
    create_transaction_tools,
)


def test_repository_filters_inclusively_and_normalizes_client_id(
    synthetic_repository,
):
    frame, period = synthetic_repository.get_transactions(
        " c003 ",
        start_date=date(2026, 10, 20),
        end_date=date(2026, 10, 22),
    )

    # The twelve designed burst transactions plus one ordinary transaction
    # generated on the same inclusive three-day window.
    assert len(frame) == 13
    assert frame["date"].min().date() == date(2026, 10, 20)
    assert frame["date"].max().date() == date(2026, 10, 22)
    assert set(frame["client_id"]) == {"C003"}
    assert period.policy == "periodo_explicito"


def test_period_resolution_is_anchored_to_dataset_not_system_clock(
    synthetic_repository,
):
    default_period = synthetic_repository.resolve_period()
    relative_period = synthetic_repository.resolve_period(last_n_days=30)

    assert default_period.reference_date == date(2026, 10, 31)
    assert (default_period.start_date, default_period.end_date) == (
        date(2026, 10, 1),
        date(2026, 10, 31),
    )
    assert (relative_period.start_date, relative_period.end_date) == (
        date(2026, 10, 2),
        date(2026, 10, 31),
    )
    assert relative_period.policy == "periodo_relativo_a_fecha_maxima_del_dataset"


def test_period_input_rejects_conflicts_and_invalid_client_id():
    with pytest.raises(ValidationError):
        ClientPeriodInput(
            client_id="C001",
            start_date="2026-10-01",
            last_n_days=30,
        )
    with pytest.raises(ValidationError):
        ClientPeriodInput(client_id="../../etc/passwd")


def test_transaction_tools_return_json_from_repository(synthetic_repository):
    get_transactions, calculate = create_transaction_tools(synthetic_repository)

    transaction_payload = json.loads(
        get_transactions.invoke(
            {
                "client_id": "C001",
                "start_date": "2026-10-01",
                "end_date": "2026-10-31",
            }
        )
    )
    risk_payload = json.loads(
        calculate.invoke(
            {
                "client_id": "C003",
                "start_date": "2026-10-01",
                "end_date": "2026-10-31",
            }
        )
    )

    assert transaction_payload["transaction_count"] == 8
    assert transaction_payload["period"]["reference_date"] == "2026-10-31"
    assert risk_payload["risk_level"] == "alto"
    assert risk_payload["risk_score"] >= 70
    assert risk_payload["verified_response_facts"]
    joined = " ".join(risk_payload["verified_response_facts"])
    assert "2026-07-01 a 2026-09-30" in joined
    assert "MXN" in joined
    assert "normalized_score=" in joined
    assert "contribution=" in joined


def test_verified_facts_do_not_turn_missing_history_into_zero(
    synthetic_repository,
):
    _, calculate = create_transaction_tools(synthetic_repository)
    payload = json.loads(
        calculate.invoke(
            {
                "client_id": "C006",
                "start_date": "2026-10-01",
                "end_date": "2026-10-31",
            }
        )
    )

    joined = " ".join(payload["verified_response_facts"])
    assert "no_disponible" in joined
    assert "score simulado: null" not in joined
    assert "score simulado: no_disponible" in joined
