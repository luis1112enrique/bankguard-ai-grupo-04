"""Deterministic transaction indicators and simulated BankGuard risk score.

Nothing in this module is a regulatory threshold or evidence of unlawful
activity.  The score is an explainable prioritisation rule for the course demo
and must be followed by human review.
"""

from __future__ import annotations

import math
import re
from datetime import date, datetime
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .config import RiskEngineConfig as RiskConfig

REQUIRED_TRANSACTION_COLUMNS = {
    "transaction_id",
    "client_id",
    "date",
    "amount",
    "currency",
    "transaction_type",
    "channel",
    "location",
}
REQUIRED_PROFILE_COLUMNS = {
    "client_id",
    "onboarding_date",
    "history_available_from",
    "typical_amount",
    "max_typical_amount",
    "expected_weekly_frequency",
    "expected_currency",
    "expected_transaction_types",
    "expected_channels",
    "expected_locations",
}
TEXT_TRANSACTION_COLUMNS = {
    "transaction_id",
    "client_id",
    "currency",
    "transaction_type",
    "channel",
    "location",
}


class DatasetValidationError(ValueError):
    """Raised when transaction or profile data violates the data contract."""


def _require_columns(
    frame: pd.DataFrame,
    required: set[str],
    dataset_name: str,
) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise DatasetValidationError(
            f"{dataset_name} is missing required columns: {', '.join(missing)}"
        )


def _blank_mask(series: pd.Series) -> pd.Series:
    return series.isna() | series.astype("string").str.strip().eq("")


def validate_transactions(
    transactions: pd.DataFrame,
    *,
    allowed_currencies: tuple[str, ...] = ("MXN", "USD"),
) -> pd.DataFrame:
    """Validate and return a normalized copy of transaction data."""

    if not isinstance(transactions, pd.DataFrame):
        raise DatasetValidationError("transactions must be a pandas DataFrame")
    _require_columns(transactions, REQUIRED_TRANSACTION_COLUMNS, "transactions")
    data = transactions.copy()

    for column in TEXT_TRANSACTION_COLUMNS:
        if _blank_mask(data[column]).any():
            raise DatasetValidationError(f"transactions.{column} cannot be blank")
        data[column] = data[column].astype("string").str.strip()
    data["client_id"] = data["client_id"].str.upper()

    if data["transaction_id"].duplicated().any():
        duplicates = sorted(
            data.loc[data["transaction_id"].duplicated(False), "transaction_id"]
            .astype(str)
            .unique()
        )
        raise DatasetValidationError(
            f"Duplicate transaction_id values: {', '.join(duplicates[:5])}"
        )

    parsed_dates = pd.to_datetime(data["date"], errors="coerce")
    if parsed_dates.isna().any():
        raise DatasetValidationError("transactions.date contains invalid dates")
    if getattr(parsed_dates.dt, "tz", None) is not None:
        parsed_dates = parsed_dates.dt.tz_convert(None)
    data["date"] = parsed_dates.dt.normalize()

    amounts = pd.to_numeric(data["amount"], errors="coerce")
    finite = np.isfinite(amounts.to_numpy(dtype=float, na_value=np.nan))
    if amounts.isna().any() or not finite.all() or (amounts <= 0).any():
        raise DatasetValidationError(
            "transactions.amount must contain finite values greater than zero"
        )
    data["amount"] = amounts.astype(float)

    data["currency"] = data["currency"].str.upper()
    invalid_currency_format = ~data["currency"].str.fullmatch(r"[A-Z]{3}")
    unsupported = ~data["currency"].isin(set(allowed_currencies))
    if invalid_currency_format.any() or unsupported.any():
        invalid = sorted(
            data.loc[invalid_currency_format | unsupported, "currency"].unique()
        )
        raise DatasetValidationError(
            f"Unsupported or invalid currency values: {', '.join(invalid)}"
        )

    return data.sort_values(["date", "transaction_id"], kind="stable").reset_index(
        drop=True
    )


def validate_client_profiles(profiles: pd.DataFrame) -> pd.DataFrame:
    """Validate and return a normalized copy of synthetic client profiles."""

    if not isinstance(profiles, pd.DataFrame):
        raise DatasetValidationError("profiles must be a pandas DataFrame")
    _require_columns(profiles, REQUIRED_PROFILE_COLUMNS, "profiles")
    data = profiles.copy()

    if _blank_mask(data["client_id"]).any():
        raise DatasetValidationError("profiles.client_id cannot be blank")
    data["client_id"] = data["client_id"].astype("string").str.strip().str.upper()
    if data["client_id"].duplicated().any():
        raise DatasetValidationError("profiles.client_id must be unique")

    for column in (
        "expected_currency",
        "expected_transaction_types",
        "expected_channels",
        "expected_locations",
    ):
        if _blank_mask(data[column]).any():
            raise DatasetValidationError(f"profiles.{column} cannot be blank")
        data[column] = data[column].astype("string").str.strip()

    for column in ("onboarding_date", "history_available_from"):
        parsed = pd.to_datetime(data[column], errors="coerce")
        if parsed.isna().any():
            raise DatasetValidationError(f"profiles.{column} contains invalid dates")
        data[column] = parsed.dt.normalize()

    for column in (
        "typical_amount",
        "max_typical_amount",
        "expected_weekly_frequency",
    ):
        values = pd.to_numeric(data[column], errors="coerce")
        finite = np.isfinite(values.to_numpy(dtype=float, na_value=np.nan))
        if values.isna().any() or not finite.all() or (values <= 0).any():
            raise DatasetValidationError(
                f"profiles.{column} must contain finite values greater than zero"
            )
        data[column] = values.astype(float)

    if (data["max_typical_amount"] < data["typical_amount"]).any():
        raise DatasetValidationError(
            "profiles.max_typical_amount cannot be smaller than typical_amount"
        )
    return data.reset_index(drop=True)


def validate_dataset_links(
    transactions: pd.DataFrame,
    profiles: pd.DataFrame,
) -> None:
    unknown = sorted(set(transactions["client_id"]) - set(profiles["client_id"]))
    if unknown:
        raise DatasetValidationError(
            f"Transactions reference unknown clients: {', '.join(unknown)}"
        )
    mixed_currency_clients = sorted(
        transactions.groupby("client_id")["currency"]
        .nunique()
        .loc[lambda counts: counts > 1]
        .index.astype(str)
        .tolist()
    )
    if mixed_currency_clients:
        raise DatasetValidationError(
            "Currency conversion is not configured; each client must use a "
            "single currency. Mixed currencies: " + ", ".join(mixed_currency_clients)
        )


def _as_day(value: str | date | datetime | pd.Timestamp, name: str) -> pd.Timestamp:
    try:
        parsed = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a valid date") from exc
    if pd.isna(parsed):
        raise ValueError(f"{name} must be a valid date")
    if parsed.tzinfo is not None:
        parsed = parsed.tz_convert(None)
    return parsed.normalize()


def _json_number(value: Any, digits: int = 6) -> float | int | None:
    if value is None or pd.isna(value):
        return None
    numeric = float(value)
    if not math.isfinite(numeric):
        return None
    rounded = round(numeric, digits)
    return int(rounded) if rounded.is_integer() else rounded


def _clip_factor(value: float) -> float:
    return float(np.clip(value, 0.0, 100.0))


def classify_risk(score: float, config: RiskConfig | None = None) -> str:
    """Classify a valid 0-100 score using exact, non-rounded boundaries."""

    settings = config or RiskConfig()
    numeric_score = float(score)
    if not math.isfinite(numeric_score) or not 0 <= numeric_score <= 100:
        raise ValueError("score must be finite and between 0 and 100")
    if numeric_score < settings.medium_threshold:
        return "bajo"
    if numeric_score < settings.high_threshold:
        return "medio"
    return "alto"


def calculate_weighted_score(
    factors: Mapping[str, float],
    config: RiskConfig | None = None,
) -> tuple[float, str, dict[str, float]]:
    """Calculate the deterministic score from four normalized factors."""

    settings = config or RiskConfig()
    missing = set(settings.weights) - set(factors)
    if missing:
        raise ValueError(f"Missing risk factors: {sorted(missing)!r}")
    extras = set(factors) - set(settings.weights)
    if extras:
        raise ValueError(f"Unknown risk factors: {sorted(extras)!r}")

    contributions: dict[str, float] = {}
    raw_score = 0.0
    for name, weight in settings.weights.items():
        value = float(factors[name])
        if not math.isfinite(value) or not 0 <= value <= 100:
            raise ValueError(f"Risk factor {name!r} must be between 0 and 100")
        contribution = value * float(weight)
        contributions[name] = round(contribution, 6)
        raw_score += contribution
    published_score = round(raw_score, 6)
    return (
        published_score,
        classify_risk(published_score, settings),
        contributions,
    )


def _parse_expected_values(value: Any) -> set[str]:
    return {
        item.strip().casefold()
        for item in re.split(r"[|,;]", str(value))
        if item.strip()
    }


def _period_metrics(
    frame: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    concentration_window_days: int,
) -> dict[str, float | int | str | None]:
    period_days = (end - start).days + 1
    count = len(frame)
    if count == 0:
        return {
            "period_days": period_days,
            "transaction_count": 0,
            "currency": None,
            "total_amount": 0.0,
            "average_amount": None,
            "maximum_amount": None,
            "standard_deviation": None,
            "active_days": 0,
            "daily_frequency": 0.0,
            "weekly_frequency": 0.0,
            "amount_hhi": None,
            "normalized_hhi": None,
            "rolling_window_amount_share_pct": None,
        }

    amounts = frame["amount"].astype(float)
    total = float(amounts.sum())
    shares = amounts / total
    hhi = float(np.square(shares).sum())
    if count == 1:
        normalized_hhi = 100.0
    else:
        minimum_hhi = 1.0 / count
        normalized_hhi = _clip_factor(100.0 * (hhi - minimum_hhi) / (1.0 - minimum_hhi))

    daily = (
        frame.groupby("date", sort=True)["amount"]
        .sum()
        .reindex(pd.date_range(start, end, freq="D"), fill_value=0.0)
    )
    rolling_share = float(
        daily.rolling(concentration_window_days, min_periods=1).sum().max()
        / total
        * 100.0
    )
    return {
        "period_days": period_days,
        "transaction_count": count,
        "currency": str(frame["currency"].iloc[0]),
        "total_amount": _json_number(total, 2),
        "average_amount": _json_number(amounts.mean(), 2),
        "maximum_amount": _json_number(amounts.max(), 2),
        "standard_deviation": _json_number(amounts.std(ddof=0), 2),
        "active_days": int(frame["date"].nunique()),
        "daily_frequency": _json_number(count / period_days),
        "weekly_frequency": _json_number(count / period_days * 7.0),
        "amount_hhi": _json_number(hhi),
        "normalized_hhi": _json_number(normalized_hhi),
        "rolling_window_amount_share_pct": _json_number(rolling_share),
    }


def _profile_factor(
    current: pd.DataFrame,
    profile: pd.Series,
    settings: RiskConfig,
) -> tuple[float, dict[str, float]]:
    typical_amount = float(profile["typical_amount"])
    amount_score = _clip_factor(
        max(0.0, float(current["amount"].mean()) / typical_amount - 1.0) * 100.0
    )

    expected_by_column = {
        "channel": _parse_expected_values(profile["expected_channels"]),
        "location": _parse_expected_values(profile["expected_locations"]),
        "transaction_type": _parse_expected_values(
            profile["expected_transaction_types"]
        ),
        "currency": _parse_expected_values(profile["expected_currency"]),
    }
    components = {"amount": amount_score}
    for column, expected in expected_by_column.items():
        observed = current[column].astype("string").str.strip().str.casefold()
        components[column] = float((~observed.isin(expected)).mean() * 100.0)

    factor = sum(
        components[name] * float(settings.profile_weights[name])
        for name in settings.profile_weights
    )
    return _clip_factor(factor), {
        name: round(value, 6) for name, value in components.items()
    }


def _outlier_threshold(profile: pd.Series, history: pd.DataFrame, z: float) -> float:
    profile_limit = float(profile["max_typical_amount"])
    if history.empty:
        return profile_limit
    amounts = history["amount"].to_numpy(dtype=float)
    median = float(np.median(amounts))
    mad = float(np.median(np.abs(amounts - median)))
    if mad == 0:
        robust_limit = median * 3.0
    else:
        robust_limit = median + z * 1.4826 * mad
    return max(profile_limit, robust_limit)


def _find_atypical_transactions(
    current: pd.DataFrame,
    history: pd.DataFrame,
    profile: pd.Series,
    history_days: int,
    settings: RiskConfig,
) -> tuple[list[dict[str, Any]], float, int]:
    amount_threshold = _outlier_threshold(profile, history, settings.outlier_mad_z)
    expected = {
        "channel": _parse_expected_values(profile["expected_channels"]),
        "location": _parse_expected_values(profile["expected_locations"]),
        "transaction_type": _parse_expected_values(
            profile["expected_transaction_types"]
        ),
        "currency": _parse_expected_values(profile["expected_currency"]),
    }
    history_daily_rate = len(history) / history_days if history_days else 0.0
    burst_threshold = max(
        settings.burst_minimum_transactions,
        math.ceil(history_daily_rate * settings.burst_rate_multiplier),
    )
    daily_counts = current.groupby("date")["transaction_id"].transform("count")

    atypical: list[dict[str, Any]] = []
    for row_number, (_, row) in enumerate(current.iterrows()):
        reasons = []
        if float(row["amount"]) > amount_threshold:
            reasons.append("monto_superior_al_umbral_robusto")
        for column, allowed in expected.items():
            if str(row[column]).strip().casefold() not in allowed:
                reasons.append(f"{column}_fuera_del_perfil")
        if int(daily_counts.iloc[row_number]) >= burst_threshold:
            reasons.append("concentracion_diaria_de_operaciones")
        if reasons:
            atypical.append(
                {
                    "transaction_id": str(row["transaction_id"]),
                    "date": pd.Timestamp(row["date"]).date().isoformat(),
                    "amount": _json_number(row["amount"], 2),
                    "currency": str(row["currency"]),
                    "reasons": reasons,
                }
            )
    return atypical, amount_threshold, burst_threshold


class RiskEngine:
    """Validate data once and calculate repeatable client-period assessments."""

    def __init__(
        self,
        transactions: pd.DataFrame,
        profiles: pd.DataFrame,
        config: RiskConfig | None = None,
    ) -> None:
        self.config = config or RiskConfig()
        self.transactions = validate_transactions(
            transactions,
            allowed_currencies=self.config.allowed_currencies,
        )
        self.profiles = validate_client_profiles(profiles)
        validate_dataset_links(self.transactions, self.profiles)

    @classmethod
    def from_csv(
        cls,
        transaction_path: Path | str,
        profile_path: Path | str,
        config: RiskConfig | None = None,
    ) -> "RiskEngine":
        return cls(
            pd.read_csv(transaction_path),
            pd.read_csv(profile_path),
            config=config,
        )

    def get_client_transactions(
        self,
        client_id: str,
        start_date: str | date | datetime | pd.Timestamp,
        end_date: str | date | datetime | pd.Timestamp,
    ) -> pd.DataFrame:
        client = str(client_id).strip().upper()
        if client not in set(self.profiles["client_id"]):
            raise KeyError(f"Unknown client_id: {client}")
        start = _as_day(start_date, "start_date")
        end = _as_day(end_date, "end_date")
        if end < start:
            raise ValueError("end_date cannot be earlier than start_date")
        if (end - start).days + 1 > self.config.maximum_analysis_days:
            raise ValueError(
                f"Analysis period cannot exceed {self.config.maximum_analysis_days} days"
            )
        mask = self.transactions["client_id"].eq(client) & self.transactions[
            "date"
        ].between(start, end, inclusive="both")
        return self.transactions.loc[mask].copy().reset_index(drop=True)

    def calculate_risk_indicators(
        self,
        client_id: str,
        start_date: str | date | datetime | pd.Timestamp,
        end_date: str | date | datetime | pd.Timestamp,
    ) -> dict[str, Any]:
        """Calculate metrics, atypical operations and a deterministic score."""

        client = str(client_id).strip().upper()
        start = _as_day(start_date, "start_date")
        end = _as_day(end_date, "end_date")
        if end < start:
            raise ValueError("end_date cannot be earlier than start_date")
        current = self.get_client_transactions(client, start, end)
        profile = self.profiles.loc[self.profiles["client_id"].eq(client)].iloc[0]

        history_end = start - pd.Timedelta(days=1)
        requested_history_start = start - pd.Timedelta(
            days=self.config.history_lookback_days
        )
        history_available_from = max(
            pd.Timestamp(profile["history_available_from"]),
            pd.Timestamp(profile["onboarding_date"]),
        )
        effective_history_start = max(
            requested_history_start,
            history_available_from,
        )
        history_days = (
            max(0, (history_end - effective_history_start).days + 1)
            if history_end >= effective_history_start
            else 0
        )
        if history_days:
            history = self.get_client_transactions(
                client,
                effective_history_start,
                history_end,
            )
        else:
            history = self.transactions.iloc[0:0].copy()

        metrics = _period_metrics(
            current,
            start,
            end,
            self.config.concentration_window_days,
        )
        historical_metrics = _period_metrics(
            history,
            effective_history_start if history_days else history_end,
            history_end,
            self.config.concentration_window_days,
        )
        historical_metrics["coverage_days"] = history_days
        historical_metrics["window_start"] = (
            effective_history_start.date().isoformat() if history_days else None
        )
        historical_metrics["window_end"] = (
            history_end.date().isoformat() if history_days else None
        )

        warnings = [
            "Puntuacion simulada para priorizar revision humana; no es un umbral regulatorio.",
            "El resultado no constituye evidencia de actividad ilicita.",
        ]
        reasons: list[str] = []
        if len(current) < self.config.minimum_analysis_transactions:
            reasons.append(
                "El periodo contiene menos de "
                f"{self.config.minimum_analysis_transactions} operaciones."
            )
        if len(history) < self.config.minimum_history_transactions:
            reasons.append(
                "El historial contiene menos de "
                f"{self.config.minimum_history_transactions} operaciones."
            )
        if history_days < self.config.minimum_history_days:
            reasons.append(
                "La cobertura historica es menor a "
                f"{self.config.minimum_history_days} dias."
            )

        atypical, amount_threshold, burst_threshold = _find_atypical_transactions(
            current,
            history,
            profile,
            history_days,
            self.config,
        )
        metrics["atypical_transaction_count"] = len(atypical)
        metrics["atypical_transaction_rate_pct"] = (
            _json_number(len(atypical) / len(current) * 100.0) if len(current) else None
        )

        factors: dict[str, float | None] = {
            "historical_deviation": None,
            "atypical_frequency": None,
            "amount_concentration": None,
            "profile_inconsistency": None,
        }
        factor_details: dict[str, Any] = {}

        if len(current):
            normalized_hhi = float(metrics["normalized_hhi"] or 0.0)
            rolling_share = float(metrics["rolling_window_amount_share_pct"] or 0.0)
            concentration = _clip_factor(max(normalized_hhi, rolling_share))
            factors["amount_concentration"] = concentration
            factor_details["amount_concentration"] = {
                "normalized_hhi": _json_number(normalized_hhi),
                "rolling_window_amount_share_pct": _json_number(rolling_share),
                "explanation": (
                    "Maximo entre HHI normalizado y porcentaje del monto concentrado "
                    f"en {self.config.concentration_window_days} dias."
                ),
            }
            profile_factor, profile_components = _profile_factor(
                current, profile, self.config
            )
            factors["profile_inconsistency"] = profile_factor
            factor_details["profile_inconsistency"] = {
                "components": profile_components,
                "explanation": "Desviaciones de monto y categorias frente al perfil.",
            }

        history_sufficient = (
            len(history) >= self.config.minimum_history_transactions
            and history_days >= self.config.minimum_history_days
        )
        if len(current) and history_sufficient:
            analysis_days = int(metrics["period_days"])
            expected_amount = (
                float(historical_metrics["total_amount"] or 0.0)
                / history_days
                * analysis_days
            )
            expected_count = len(history) / history_days * analysis_days

            if expected_amount > 0:
                amount_variation = (
                    float(metrics["total_amount"] or 0.0) / expected_amount - 1.0
                ) * 100.0
                factors["historical_deviation"] = _clip_factor(abs(amount_variation))
            else:
                amount_variation = None
                reasons.append("El monto historico esperado es cero.")

            if expected_count > 0:
                frequency_variation = (len(current) / expected_count - 1.0) * 100.0
                factors["atypical_frequency"] = _clip_factor(abs(frequency_variation))
            else:
                frequency_variation = None
                reasons.append("La frecuencia historica esperada es cero.")

            metrics["expected_amount_from_history"] = _json_number(expected_amount, 2)
            metrics["amount_variation_pct"] = _json_number(amount_variation)
            metrics["expected_transaction_count_from_history"] = _json_number(
                expected_count
            )
            metrics["frequency_variation_pct"] = _json_number(frequency_variation)
            factor_details["historical_deviation"] = {
                "expected_amount": _json_number(expected_amount, 2),
                "variation_pct": _json_number(amount_variation),
                "explanation": (
                    "Variacion absoluta del monto por dia frente al historial."
                ),
            }
            factor_details["atypical_frequency"] = {
                "expected_transaction_count": _json_number(expected_count),
                "variation_pct": _json_number(frequency_variation),
                "explanation": (
                    "Variacion absoluta de la frecuencia diaria frente al historial."
                ),
            }
        else:
            metrics["expected_amount_from_history"] = None
            metrics["amount_variation_pct"] = None
            metrics["expected_transaction_count_from_history"] = None
            metrics["frequency_variation_pct"] = None

        conclusive = (
            len(current) >= self.config.minimum_analysis_transactions
            and history_sufficient
            and all(value is not None for value in factors.values())
        )
        if conclusive:
            score, level, contributions = calculate_weighted_score(
                {name: float(value) for name, value in factors.items()},
                self.config,
            )
            status = "concluyente"
        else:
            score = None
            level = "no_concluyente"
            contributions = {
                name: (
                    round(float(value) * float(self.config.weights[name]), 6)
                    if value is not None
                    else None
                )
                for name, value in factors.items()
            }
            status = "sin_datos" if current.empty else "no_concluyente"
            warnings.append(
                "No se asigno nivel bajo porque falta evidencia suficiente."
            )

        for name, value in factors.items():
            detail = factor_details.setdefault(name, {})
            detail.update(
                {
                    "normalized_score": _json_number(value),
                    "weight": float(self.config.weights[name]),
                    "contribution": _json_number(contributions[name]),
                }
            )

        return {
            "client_id": client,
            "period": {
                "start_date": start.date().isoformat(),
                "end_date": end.date().isoformat(),
                "days": int(metrics["period_days"]),
            },
            "assessment_status": status,
            "metrics": metrics,
            "historical_metrics": historical_metrics,
            "risk_score": _json_number(score),
            "risk_level": level,
            "risk_factors": factor_details,
            "atypical_transactions": atypical,
            "detection_parameters": {
                "amount_threshold": _json_number(amount_threshold, 2),
                "burst_daily_transaction_threshold": burst_threshold,
            },
            "insufficiency_reasons": reasons,
            "warnings": warnings,
        }


def calculate_risk_indicators(
    transactions: pd.DataFrame,
    profiles: pd.DataFrame,
    client_id: str,
    start_date: str | date | datetime | pd.Timestamp,
    end_date: str | date | datetime | pd.Timestamp,
    config: RiskConfig | None = None,
) -> dict[str, Any]:
    """Convenience wrapper for one stateless assessment."""

    return RiskEngine(transactions, profiles, config=config).calculate_risk_indicators(
        client_id,
        start_date,
        end_date,
    )
