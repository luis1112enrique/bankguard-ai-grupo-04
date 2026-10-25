from __future__ import annotations

import pandas as pd
import pytest

from agents.bankguard.risk_engine import (
    DatasetValidationError,
    RiskEngine,
    calculate_weighted_score,
    classify_risk,
    validate_dataset_links,
    validate_transactions,
)


def _exact_engine() -> RiskEngine:
    history_dates = [
        "2026-07-01",
        "2026-07-15",
        "2026-08-01",
        "2026-08-15",
        "2026-09-01",
        "2026-09-10",
        "2026-09-20",
        "2026-09-30",
    ]
    rows = [
        {
            "transaction_id": f"H{number}",
            "client_id": "C100",
            "date": day,
            "amount": 100,
            "currency": "MXN",
            "transaction_type": "transferencia",
            "channel": "web",
            "location": "CDMX",
        }
        for number, day in enumerate(history_dates, start=1)
    ]
    rows.extend(
        {
            "transaction_id": f"C{number}",
            "client_id": "C100",
            "date": day,
            "amount": amount,
            "currency": "MXN",
            "transaction_type": "transferencia",
            "channel": "web",
            "location": "CDMX",
        }
        for number, (day, amount) in enumerate(
            [
                ("2026-10-01", 100),
                ("2026-10-02", 200),
                ("2026-10-02", 300),
                ("2026-10-10", 400),
            ],
            start=1,
        )
    )
    profile = pd.DataFrame(
        [
            {
                "client_id": "C100",
                "onboarding_date": "2026-07-01",
                "history_available_from": "2026-07-01",
                "typical_amount": 100,
                "max_typical_amount": 250,
                "expected_weekly_frequency": 1,
                "expected_currency": "MXN",
                "expected_transaction_types": "transferencia",
                "expected_channels": "web",
                "expected_locations": "CDMX",
            }
        ]
    )
    return RiskEngine(pd.DataFrame(rows), profile)


def test_metrics_and_weighted_score_are_exact():
    result = _exact_engine().calculate_risk_indicators(
        "C100", "2026-10-01", "2026-10-10"
    )
    metrics = result["metrics"]

    assert metrics == {
        "period_days": 10,
        "transaction_count": 4,
        "currency": "MXN",
        "total_amount": 1000,
        "average_amount": 250,
        "maximum_amount": 400,
        "standard_deviation": 111.8,
        "active_days": 3,
        "daily_frequency": 0.4,
        "weekly_frequency": 2.8,
        "amount_hhi": 0.3,
        "normalized_hhi": 6.666667,
        "rolling_window_amount_share_pct": 60,
        "atypical_transaction_count": 1,
        "atypical_transaction_rate_pct": 25,
        "expected_amount_from_history": 86.96,
        "amount_variation_pct": 1050,
        "expected_transaction_count_from_history": 0.869565,
        "frequency_variation_pct": 360,
    }
    assert result["risk_score"] == 83
    assert result["risk_level"] == "alto"
    contributions = [
        factor["contribution"] for factor in result["risk_factors"].values()
    ]
    assert sum(contributions) == pytest.approx(result["risk_score"])
    assert result["atypical_transactions"][0]["transaction_id"] == "C4"


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0, "bajo"),
        (39.999999, "bajo"),
        (40, "medio"),
        (69.999999, "medio"),
        (70, "alto"),
        (100, "alto"),
    ],
)
def test_risk_boundaries(score, expected):
    assert classify_risk(score) == expected


def test_weight_formula_uses_40_25_20_15():
    score, level, contributions = calculate_weighted_score(
        {
            "historical_deviation": 100,
            "atypical_frequency": 0,
            "amount_concentration": 50,
            "profile_inconsistency": 100,
        }
    )

    assert score == 65
    assert level == "medio"
    assert contributions == {
        "historical_deviation": 40,
        "atypical_frequency": 0,
        "amount_concentration": 10,
        "profile_inconsistency": 15,
    }


@pytest.mark.parametrize(
    ("factor_value", "expected_score", "expected_level"),
    [
        (39.9999994, 39.999999, "bajo"),
        (39.9999996, 40.0, "medio"),
        (69.9999994, 69.999999, "medio"),
        (69.9999996, 70.0, "alto"),
    ],
)
def test_weighted_score_level_matches_published_boundary(
    factor_value,
    expected_score,
    expected_level,
):
    score, level, _ = calculate_weighted_score(
        {
            "historical_deviation": factor_value,
            "atypical_frequency": factor_value,
            "amount_concentration": factor_value,
            "profile_inconsistency": factor_value,
        }
    )

    assert score == expected_score
    assert level == expected_level
    assert classify_risk(score) == level


def test_published_scenarios_include_high_and_inconclusive(synthetic_engine):
    high = synthetic_engine.calculate_risk_indicators(
        "C003", "2026-10-01", "2026-10-31"
    )
    inconclusive = synthetic_engine.calculate_risk_indicators(
        "C006", "2026-10-01", "2026-10-31"
    )

    assert high["assessment_status"] == "concluyente"
    assert high["risk_level"] == "alto"
    assert high["risk_score"] >= 70
    assert inconclusive["assessment_status"] == "no_concluyente"
    assert inconclusive["risk_level"] == "no_concluyente"
    assert inconclusive["risk_score"] is None
    assert any(
        "No se asigno nivel bajo" in warning for warning in inconclusive["warnings"]
    )


def test_dataset_validation_rejects_bad_amount_and_unknown_client(synthetic_frames):
    transactions, profiles = synthetic_frames
    invalid = transactions.head(2).copy()
    invalid.loc[0, "amount"] = 0
    with pytest.raises(DatasetValidationError, match="greater than zero"):
        validate_transactions(invalid)

    unknown = transactions.head(1).copy()
    unknown.loc[:, "client_id"] = "UNKNOWN"
    with pytest.raises(DatasetValidationError, match="unknown clients"):
        validate_dataset_links(validate_transactions(unknown), profiles)


def test_dataset_validation_rejects_missing_blank_nan_and_mixed_currency(
    synthetic_frames,
):
    transactions, profiles = synthetic_frames

    with pytest.raises(DatasetValidationError, match="missing required columns"):
        validate_transactions(transactions.drop(columns=["channel"]))

    blank = transactions.head(1).copy()
    blank.loc[:, "channel"] = " "
    with pytest.raises(DatasetValidationError, match="channel cannot be blank"):
        validate_transactions(blank)

    nan_amount = transactions.head(1).copy()
    nan_amount.loc[:, "amount"] = float("nan")
    with pytest.raises(DatasetValidationError, match="finite values"):
        validate_transactions(nan_amount)

    mixed = transactions.copy()
    target = mixed.index[mixed["client_id"].eq("C001")][0]
    mixed.loc[target, "currency"] = "USD"
    with pytest.raises(DatasetValidationError, match="Mixed currencies: C001"):
        validate_dataset_links(validate_transactions(mixed), profiles)
