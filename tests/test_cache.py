import numpy as np
import pytest

from atlas_posthoc.io.cache import (
    CacheError,
    load_pipeline_observable_cache,
    read_json,
    read_observable_runs,
    write_json,
    write_observable_runs,
    write_pipeline_cache,
)


class FakeChainResult:
    def __init__(self, steps, names):
        self.steps = steps
        self.names = names
        self.map_indices = list(range(len(steps)))


class FakeObservables:
    def __init__(self):
        self.vote_runs = {
            "run5": np.array([[0.1, 0.6], [0.2, 0.7]]),
            "run6": np.array([[0.1, 0.6], [0.3, 0.8]]),
        }
        self.compactness_runs = {
            "run5": np.array([[10.0, 20.0], [11.0, 21.0]]),
            "run6": np.array([[10.5, 20.5], [11.5, 21.5]]),
        }
        self.chains = {
            "run5": FakeChainResult([0, 10], ["step0", "step10"]),
            "run6": FakeChainResult([0, 10], ["step0", "step10"]),
        }

    def summaries(self):
        return {
            "run5": {"chain_id": "run5", "n_records_processed": 2},
            "run6": {"chain_id": "run6", "n_records_processed": 2},
        }


def test_write_and_read_json(tmp_path):
    path = tmp_path / "summary.json"
    obj = {
        "x": np.float64(1.5),
        "y": np.array([1, 2, 3]),
        "z": float("nan"),
    }

    write_json(obj, path)
    loaded = read_json(path)

    assert loaded["x"] == 1.5
    assert loaded["y"] == [1, 2, 3]
    assert loaded["z"] is None


def test_write_and_read_observable_runs(tmp_path):
    runs = {
        "run5": np.array([[0.1, 0.6], [0.2, 0.7]]),
        "run6": np.array([[0.1, 0.6], [0.3, 0.8]]),
    }

    path = tmp_path / "vote_runs.parquet"

    write_observable_runs(
        runs,
        path,
        value_col="dem_share",
        steps={
            "run5": [0, 10],
            "run6": [0, 10],
        },
        names={
            "run5": ["step0", "step10"],
            "run6": ["step0", "step10"],
        },
    )

    loaded = read_observable_runs(path, value_col="dem_share")

    assert set(loaded.keys()) == {"run5", "run6"}
    assert np.allclose(loaded["run5"], runs["run5"])
    assert np.allclose(loaded["run6"], runs["run6"])


def test_write_observable_runs_rejects_bad_shape(tmp_path):
    runs = {
        "run5": np.array([0.1, 0.2, 0.3]),
    }

    with pytest.raises(CacheError):
        write_observable_runs(
            runs,
            tmp_path / "bad.parquet",
            value_col="dem_share",
        )


def test_write_pipeline_cache_and_load(tmp_path):
    obs = FakeObservables()

    manifest = write_pipeline_cache(
        tmp_path,
        observables=obs,
        vote_rhat={"max_rhat": 1.01},
        compactness_rhat={"max_rhat": 1.02},
        extra={"note": "test"},
    )

    assert (tmp_path / "manifest.json").exists()
    assert manifest.files["vote_runs"].exists()
    assert manifest.files["compactness_runs"].exists()
    assert manifest.files["chain_summaries"].exists()

    loaded = load_pipeline_observable_cache(tmp_path)

    assert set(loaded["vote_runs"].keys()) == {"run5", "run6"}
    assert set(loaded["compactness_runs"].keys()) == {"run5", "run6"}
    assert np.allclose(loaded["vote_runs"]["run5"], obs.vote_runs["run5"])
    assert np.allclose(
        loaded["compactness_runs"]["run6"],
        obs.compactness_runs["run6"],
    )

    assert loaded["summaries"]["run5"]["chain_id"] == "run5"


def test_load_pipeline_observable_cache_missing_vote_raises(tmp_path):
    with pytest.raises(CacheError):
        load_pipeline_observable_cache(tmp_path)