"""Generate the reproducible, entirely fictional BankGuard demo dataset.

The generator deliberately creates several behaviours that are useful for the
course demo.  The scenario labels live only in the client profile file; the
risk engine never uses them to calculate a score.

Default dataset
---------------
* Seed: ``20261020``.
* History window: 2026-07-01 through 2026-09-30.
* Analysis window: 2026-10-01 through 2026-10-31.
* 202 transactions across six fictional clients.

Run this module only when the demo CSV files need to be created::

    uv run python -m agents.bankguard.data_generator

Existing files are not overwritten unless ``--overwrite`` is passed.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

DEFAULT_SEED = 20261020
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data"
TRANSACTIONS_FILENAME = "bank_transactions.csv"
PROFILES_FILENAME = "client_profiles.csv"

TRANSACTION_COLUMNS = [
    "transaction_id",
    "client_id",
    "date",
    "amount",
    "currency",
    "transaction_type",
    "channel",
    "location",
]

PROFILE_COLUMNS = [
    "client_id",
    "client_name",
    "segment",
    "onboarding_date",
    "history_available_from",
    "typical_amount",
    "max_typical_amount",
    "expected_weekly_frequency",
    "expected_currency",
    "expected_transaction_types",
    "expected_channels",
    "expected_locations",
    "scenario_description",
]


@dataclass(frozen=True)
class ClientRecipe:
    """Parameters used only to produce one synthetic client."""

    client_id: str
    client_name: str
    segment: str
    onboarding_date: str
    history_available_from: str
    typical_amount: float
    max_typical_amount: float
    expected_weekly_frequency: float
    expected_currency: str
    expected_transaction_types: tuple[str, ...]
    expected_channels: tuple[str, ...]
    expected_locations: tuple[str, ...]
    historical_count: int
    october_count: int
    scenario_description: str


CLIENT_RECIPES: tuple[ClientRecipe, ...] = (
    ClientRecipe(
        client_id="C001",
        client_name="Cliente Ficticio 001",
        segment="personas",
        onboarding_date="2024-02-12",
        history_available_from="2026-07-01",
        typical_amount=2_500.0,
        max_typical_amount=7_500.0,
        expected_weekly_frequency=1.8,
        expected_currency="MXN",
        expected_transaction_types=("transferencia", "pago"),
        expected_channels=("movil", "web"),
        expected_locations=("Ciudad de Mexico",),
        historical_count=24,
        october_count=8,
        scenario_description="Comportamiento habitual estable",
    ),
    ClientRecipe(
        client_id="C002",
        client_name="Cliente Ficticio 002",
        segment="pyme",
        onboarding_date="2023-08-04",
        history_available_from="2026-07-01",
        typical_amount=6_000.0,
        max_typical_amount=18_000.0,
        expected_weekly_frequency=2.3,
        expected_currency="MXN",
        expected_transaction_types=("transferencia", "deposito"),
        expected_channels=("web", "sucursal"),
        expected_locations=("Guadalajara",),
        historical_count=30,
        october_count=10,
        scenario_description="Incremento moderado de montos en octubre",
    ),
    ClientRecipe(
        client_id="C003",
        client_name="Cliente Ficticio 003",
        segment="personas",
        onboarding_date="2024-11-20",
        history_available_from="2026-07-01",
        typical_amount=3_000.0,
        max_typical_amount=9_000.0,
        expected_weekly_frequency=1.8,
        expected_currency="MXN",
        expected_transaction_types=("transferencia", "pago"),
        expected_channels=("movil", "web"),
        expected_locations=("Ciudad de Mexico",),
        historical_count=24,
        october_count=28,
        scenario_description=(
            "Aumento fuerte de montos y operaciones concentradas fuera del perfil"
        ),
    ),
    ClientRecipe(
        client_id="C004",
        client_name="Cliente Ficticio 004",
        segment="personas",
        onboarding_date="2025-01-08",
        history_available_from="2026-07-01",
        typical_amount=1_200.0,
        max_typical_amount=4_000.0,
        expected_weekly_frequency=1.4,
        expected_currency="MXN",
        expected_transaction_types=("transferencia", "pago"),
        expected_channels=("movil",),
        expected_locations=("Puebla",),
        historical_count=18,
        october_count=26,
        scenario_description="Frecuencia atipica y rafaga de operaciones",
    ),
    ClientRecipe(
        client_id="C005",
        client_name="Cliente Ficticio 005",
        segment="pyme",
        onboarding_date="2023-05-15",
        history_available_from="2026-07-01",
        typical_amount=8_000.0,
        max_typical_amount=24_000.0,
        expected_weekly_frequency=1.5,
        expected_currency="MXN",
        expected_transaction_types=("transferencia", "deposito"),
        expected_channels=("web", "sucursal"),
        expected_locations=("Monterrey",),
        historical_count=20,
        october_count=10,
        scenario_description="Concentracion por una operacion de monto elevado",
    ),
    ClientRecipe(
        client_id="C006",
        client_name="Cliente Ficticio 006",
        segment="personas",
        onboarding_date="2026-10-15",
        history_available_from="2026-10-15",
        typical_amount=2_500.0,
        max_typical_amount=7_500.0,
        expected_weekly_frequency=1.5,
        expected_currency="MXN",
        expected_transaction_types=("transferencia", "pago"),
        expected_channels=("movil",),
        expected_locations=("Merida",),
        historical_count=0,
        october_count=4,
        scenario_description="Cliente nuevo con historial insuficiente",
    ),
)


def _sample_dates(
    rng: np.random.Generator,
    start: str,
    end: str,
    count: int,
) -> list[pd.Timestamp]:
    dates = pd.date_range(start, end, freq="D")
    if count > len(dates):
        raise ValueError("Cannot sample unique dates: count exceeds available days")
    selected = rng.choice(dates.to_numpy(), size=count, replace=False)
    return sorted(pd.Timestamp(value) for value in selected)


def _normal_amounts(
    rng: np.random.Generator,
    typical_amount: float,
    count: int,
    multiplier: float = 1.0,
) -> np.ndarray:
    values = rng.lognormal(
        mean=np.log(typical_amount * multiplier),
        sigma=0.22,
        size=count,
    )
    return np.round(
        np.clip(
            values,
            typical_amount * 0.35,
            typical_amount * multiplier * 2.2,
        ),
        2,
    )


def _normal_rows(
    rng: np.random.Generator,
    recipe: ClientRecipe,
    dates: Iterable[pd.Timestamp],
    amounts: Iterable[float],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for date, amount in zip(dates, amounts, strict=True):
        rows.append(
            {
                "client_id": recipe.client_id,
                "date": pd.Timestamp(date),
                "amount": round(float(amount), 2),
                "currency": recipe.expected_currency,
                "transaction_type": str(rng.choice(recipe.expected_transaction_types)),
                "channel": str(rng.choice(recipe.expected_channels)),
                "location": str(rng.choice(recipe.expected_locations)),
            }
        )
    return rows


def _profile_frame() -> pd.DataFrame:
    rows = []
    for recipe in CLIENT_RECIPES:
        rows.append(
            {
                "client_id": recipe.client_id,
                "client_name": recipe.client_name,
                "segment": recipe.segment,
                "onboarding_date": recipe.onboarding_date,
                "history_available_from": recipe.history_available_from,
                "typical_amount": recipe.typical_amount,
                "max_typical_amount": recipe.max_typical_amount,
                "expected_weekly_frequency": recipe.expected_weekly_frequency,
                "expected_currency": recipe.expected_currency,
                "expected_transaction_types": "|".join(
                    recipe.expected_transaction_types
                ),
                "expected_channels": "|".join(recipe.expected_channels),
                "expected_locations": "|".join(recipe.expected_locations),
                "scenario_description": recipe.scenario_description,
            }
        )
    return pd.DataFrame(rows, columns=PROFILE_COLUMNS)


def generate_synthetic_data(
    seed: int = DEFAULT_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return deterministic transaction and client-profile data frames.

    No personal data is used.  The returned transaction frame contains exactly
    202 rows with ISO-formatted dates and stable transaction IDs.
    """

    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []

    for recipe in CLIENT_RECIPES:
        if recipe.historical_count:
            rows.extend(
                _normal_rows(
                    rng,
                    recipe,
                    _sample_dates(
                        rng,
                        "2026-07-01",
                        "2026-09-30",
                        recipe.historical_count,
                    ),
                    _normal_amounts(
                        rng,
                        recipe.typical_amount,
                        recipe.historical_count,
                    ),
                )
            )

        if recipe.client_id == "C003":
            normal_count = 16
            rows.extend(
                _normal_rows(
                    rng,
                    recipe,
                    _sample_dates(rng, "2026-10-01", "2026-10-31", normal_count),
                    _normal_amounts(rng, recipe.typical_amount, normal_count),
                )
            )
            burst_dates = [
                pd.Timestamp(year=2026, month=10, day=day)
                for day in (20, 20, 20, 20, 21, 21, 21, 21, 22, 22, 22, 22)
            ]
            for index, (date, amount) in enumerate(
                zip(
                    burst_dates,
                    rng.uniform(18_000.0, 42_000.0, size=12),
                    strict=True,
                )
            ):
                rows.append(
                    {
                        "client_id": recipe.client_id,
                        "date": date,
                        "amount": round(float(amount), 2),
                        "currency": "MXN",
                        "transaction_type": "deposito_efectivo",
                        "channel": "sucursal",
                        "location": "Monterrey" if index % 2 == 0 else "Tijuana",
                    }
                )
        elif recipe.client_id == "C004":
            normal_count = 14
            rows.extend(
                _normal_rows(
                    rng,
                    recipe,
                    _sample_dates(rng, "2026-10-01", "2026-10-31", normal_count),
                    _normal_amounts(rng, recipe.typical_amount, normal_count),
                )
            )
            burst_dates = [
                pd.Timestamp(year=2026, month=10, day=day)
                for day in (12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14)
            ]
            rows.extend(
                _normal_rows(
                    rng,
                    recipe,
                    burst_dates,
                    _normal_amounts(rng, recipe.typical_amount, 12),
                )
            )
        elif recipe.client_id == "C005":
            normal_count = 9
            rows.extend(
                _normal_rows(
                    rng,
                    recipe,
                    _sample_dates(rng, "2026-10-01", "2026-10-31", normal_count),
                    _normal_amounts(rng, recipe.typical_amount, normal_count),
                )
            )
            rows.append(
                {
                    "client_id": recipe.client_id,
                    "date": pd.Timestamp("2026-10-18"),
                    "amount": 120_000.0,
                    "currency": "MXN",
                    "transaction_type": "transferencia",
                    "channel": "web",
                    "location": "Monterrey",
                }
            )
        else:
            october_start = "2026-10-15" if recipe.client_id == "C006" else "2026-10-01"
            amount_multiplier = 2.0 if recipe.client_id == "C002" else 1.0
            rows.extend(
                _normal_rows(
                    rng,
                    recipe,
                    _sample_dates(
                        rng,
                        october_start,
                        "2026-10-31",
                        recipe.october_count,
                    ),
                    _normal_amounts(
                        rng,
                        recipe.typical_amount,
                        recipe.october_count,
                        multiplier=amount_multiplier,
                    ),
                )
            )

    transactions = pd.DataFrame(rows)
    transactions = transactions.sort_values(
        ["date", "client_id", "amount"], kind="stable"
    ).reset_index(drop=True)
    transactions.insert(
        0,
        "transaction_id",
        [f"TX-2026-{number:04d}" for number in range(1, len(transactions) + 1)],
    )
    transactions["date"] = transactions["date"].dt.strftime("%Y-%m-%d")
    transactions = transactions[TRANSACTION_COLUMNS]

    profiles = _profile_frame()
    _validate_generated_frames(transactions, profiles)
    return transactions, profiles


def _validate_generated_frames(
    transactions: pd.DataFrame,
    profiles: pd.DataFrame,
) -> None:
    """Fail fast when a change breaks the published demo data contract."""

    if list(transactions.columns) != TRANSACTION_COLUMNS:
        raise ValueError("Unexpected synthetic transaction columns")
    if list(profiles.columns) != PROFILE_COLUMNS:
        raise ValueError("Unexpected synthetic profile columns")
    if len(transactions) != 202:
        raise ValueError(f"Expected 202 transactions, found {len(transactions)}")
    if transactions["transaction_id"].duplicated().any():
        raise ValueError("Synthetic transaction IDs must be unique")

    dates = pd.to_datetime(transactions["date"], errors="coerce")
    amounts = pd.to_numeric(transactions["amount"], errors="coerce")
    if dates.isna().any():
        raise ValueError("Synthetic dataset contains invalid dates")
    if amounts.isna().any() or not np.isfinite(amounts).all() or (amounts <= 0).any():
        raise ValueError("Synthetic amounts must be finite and greater than zero")

    profile_ids = set(profiles["client_id"])
    if set(transactions["client_id"]) != profile_ids:
        raise ValueError("Every synthetic client must have transactions and a profile")

    october = transactions[dates.between("2026-10-01", "2026-10-31")]
    october_counts = october.groupby("client_id").size().to_dict()
    expected_counts = {
        recipe.client_id: recipe.october_count for recipe in CLIENT_RECIPES
    }
    if october_counts != expected_counts:
        raise ValueError(f"Unexpected October scenario counts: {october_counts!r}")

    c003 = october[october["client_id"] == "C003"]
    c003_large = c003[pd.to_numeric(c003["amount"]) >= 18_000.0]
    if len(c003_large) != 12:
        raise ValueError("C003 must contain 12 clearly elevated October operations")

    c006_history = transactions[
        (transactions["client_id"] == "C006") & (dates < "2026-10-01")
    ]
    if not c006_history.empty:
        raise ValueError("C006 must not have transactions before October 2026")


def write_synthetic_data(
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    *,
    seed: int = DEFAULT_SEED,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Create both demo CSV files and return their paths.

    The operation is deliberately non-destructive by default.
    """

    destination = Path(output_dir)
    transaction_path = destination / TRANSACTIONS_FILENAME
    profile_path = destination / PROFILES_FILENAME
    existing = [path for path in (transaction_path, profile_path) if path.exists()]
    if existing and not overwrite:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(
            f"Refusing to overwrite existing synthetic data: {names}. "
            "Pass overwrite=True only when replacement is intentional."
        )

    transactions, profiles = generate_synthetic_data(seed=seed)
    destination.mkdir(parents=True, exist_ok=True)
    transactions.to_csv(transaction_path, index=False, encoding="utf-8")
    profiles.to_csv(profile_path, index=False, encoding="utf-8")
    return transaction_path, profile_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    transaction_path, profile_path = write_synthetic_data(
        args.output_dir,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(f"Created {transaction_path}")
    print(f"Created {profile_path}")


if __name__ == "__main__":
    main()
