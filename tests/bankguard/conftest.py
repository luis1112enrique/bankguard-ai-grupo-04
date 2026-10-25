"""Shared, service-free fixtures for the BankGuard test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.bankguard.config import BankGuardSettings, load_risk_config
from agents.bankguard.data_generator import generate_synthetic_data
from agents.bankguard.risk_engine import RiskEngine
from agents.bankguard.transaction_tools import BankingDataRepository


@pytest.fixture(scope="session")
def synthetic_frames():
    transactions, profiles = generate_synthetic_data()
    return transactions, profiles


@pytest.fixture(scope="session")
def synthetic_engine(synthetic_frames) -> RiskEngine:
    transactions, profiles = synthetic_frames
    return RiskEngine(
        transactions.copy(),
        profiles.copy(),
        config=load_risk_config(),
    )


@pytest.fixture()
def synthetic_repository(tmp_path, synthetic_frames) -> BankingDataRepository:
    transactions, profiles = synthetic_frames
    transaction_path = tmp_path / "bank_transactions.csv"
    profile_path = tmp_path / "client_profiles.csv"
    transactions.to_csv(transaction_path, index=False)
    profiles.to_csv(profile_path, index=False)
    return BankingDataRepository(
        transaction_path,
        profile_path,
        risk_config=load_risk_config(),
    )


@pytest.fixture()
def isolated_settings(tmp_path) -> BankGuardSettings:
    data_dir = tmp_path / "data"
    outputs_dir = tmp_path / "outputs"
    data_dir.mkdir()
    return BankGuardSettings(
        project_root=tmp_path,
        transactions_csv=data_dir / "bank_transactions.csv",
        profiles_csv=data_dir / "client_profiles.csv",
        outputs_dir=outputs_dir,
        charts_dir=outputs_dir / "charts",
        alerts_dir=outputs_dir / "alerts",
        risk_config_path=PROJECT_ROOT / "agents" / "bankguard" / "risk_config.json",
        index_name="unit-test-index",
    )
