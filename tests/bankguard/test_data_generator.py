from __future__ import annotations

import pandas as pd
import pytest

from agents.bankguard.data_generator import (
    PROFILE_COLUMNS,
    TRANSACTION_COLUMNS,
    generate_synthetic_data,
    write_synthetic_data,
)


def test_generator_is_reproducible_and_has_published_shape():
    transactions_a, profiles_a = generate_synthetic_data(seed=20261020)
    transactions_b, profiles_b = generate_synthetic_data(seed=20261020)

    pd.testing.assert_frame_equal(transactions_a, transactions_b)
    pd.testing.assert_frame_equal(profiles_a, profiles_b)
    assert len(transactions_a) == 202
    assert len(profiles_a) == 6
    assert list(transactions_a.columns) == TRANSACTION_COLUMNS
    assert list(profiles_a.columns) == PROFILE_COLUMNS
    assert transactions_a["transaction_id"].is_unique
    assert set(transactions_a["client_id"]) == {
        "C001",
        "C002",
        "C003",
        "C004",
        "C005",
        "C006",
    }
    assert profiles_a["client_name"].str.contains("Ficticio").all()


def test_different_seed_changes_transactions_but_not_data_contract():
    transactions_a, profiles_a = generate_synthetic_data(seed=1)
    transactions_b, profiles_b = generate_synthetic_data(seed=2)

    assert not transactions_a.equals(transactions_b)
    pd.testing.assert_frame_equal(profiles_a, profiles_b)
    assert len(transactions_a) == len(transactions_b) == 202


def test_writer_is_non_destructive_by_default(tmp_path):
    transaction_path, profile_path = write_synthetic_data(tmp_path)

    assert len(pd.read_csv(transaction_path)) == 202
    assert len(pd.read_csv(profile_path)) == 6
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_synthetic_data(tmp_path)
