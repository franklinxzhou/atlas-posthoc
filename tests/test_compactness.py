import pytest

from atlas_posthoc.observables.compactness import (
    CompactnessError,
    extract_compactness_scores_from_data,
    extract_ordered_compactness_from_data,
    ordered_compactness_from_scores,
)


class DummyRecord:
    def __init__(self, data):
        self.data = data


def test_extract_compactness_scores_from_default_key():
    data = {
        "get_isoperimetric_scores": [12.0, 9.5, 14.2],
    }

    scores, source_key = extract_compactness_scores_from_data(data)

    assert scores == [12.0, 9.5, 14.2]
    assert source_key == "get_isoperimetric_scores"


def test_extract_compactness_scores_from_singular_key():
    data = {
        "get_isoperimetric_score": [12.0, 9.5, 14.2],
    }

    scores, source_key = extract_compactness_scores_from_data(data)

    assert scores == [12.0, 9.5, 14.2]
    assert source_key == "get_isoperimetric_score"


def test_ordered_compactness_ascending():
    result = ordered_compactness_from_scores(
        [12.0, 9.5, 14.2],
        n_districts=3,
    )

    assert result.ordered_scores == [9.5, 12.0, 14.2]
    assert result.raw_scores == [12.0, 9.5, 14.2]
    assert result.n_districts_observed == 3
    assert result.missing_score_count == 0


def test_ordered_compactness_descending():
    result = ordered_compactness_from_scores(
        [12.0, 9.5, 14.2],
        n_districts=3,
        sort="descending",
    )

    assert result.ordered_scores == [14.2, 12.0, 9.5]


def test_ordered_compactness_no_sort():
    result = ordered_compactness_from_scores(
        [12.0, 9.5, 14.2],
        n_districts=3,
        sort="none",
    )

    assert result.ordered_scores == [12.0, 9.5, 14.2]


def test_extract_ordered_compactness_from_data():
    data = {
        "get_isoperimetric_scores": [12.0, 9.5, 14.2],
    }

    result = extract_ordered_compactness_from_data(
        data,
        n_districts=3,
    )

    assert result.ordered_scores == [9.5, 12.0, 14.2]
    assert result.source_key == "get_isoperimetric_scores"


def test_dict_payload_sorted_by_district_label():
    data = {
        "get_isoperimetric_scores": {
            "3": 14.2,
            "1": 12.0,
            "2": 9.5,
        }
    }

    scores, source_key = extract_compactness_scores_from_data(data)

    assert source_key == "get_isoperimetric_scores"
    assert scores == [12.0, 9.5, 14.2]


def test_missing_compactness_key_raises():
    data = {
        "other_metric": [1, 2, 3],
    }

    with pytest.raises(CompactnessError):
        extract_compactness_scores_from_data(data)


def test_wrong_number_of_scores_raises():
    with pytest.raises(CompactnessError):
        ordered_compactness_from_scores(
            [12.0, 9.5],
            n_districts=3,
        )


def test_missing_score_raises_by_default():
    with pytest.raises(CompactnessError):
        ordered_compactness_from_scores(
            [12.0, None, 14.2],
            n_districts=3,
        )


def test_missing_score_allowed():
    result = ordered_compactness_from_scores(
        [12.0, None, 14.2],
        n_districts=3,
        allow_missing_scores=True,
    )

    assert result.ordered_scores == [12.0, 14.2, None]
    assert result.missing_score_count == 1