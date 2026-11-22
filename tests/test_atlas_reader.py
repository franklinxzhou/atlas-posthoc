import gzip
import json

import pytest

from atlas_posthoc.io.atlas_reader import AtlasRecordError, read_atlas_records


@pytest.fixture(params=[".jsonl", ".jsonl.gz"], ids=["plain", "gzip"])
def atlas_path(tmp_path, request):
    path = tmp_path / f"toy-chain{request.param}"
    first_map = {
        "name": "step0",
        "weight": 2,
        "data": {"get_isoperimetric_scores": [10, 20]},
        "districting": [{'["North"]': 1}, {'["South", "001"]': 2}],
    }
    second_map = {
        "name": "step-50",
        "districting": {'["North"]': 2, '["South", "001"]': 1},
    }
    third_map = {**first_map, "name": "step_100"}
    content = "\n".join(
        [
            json.dumps("This is an Atlas for Redistricting Maps"),
            json.dumps({"description": "Synthetic multiscale chain"}),
            json.dumps({"districts": 2, "levels in graph": ["county", "prec_id"]}),
            json.dumps(first_map),
            json.dumps(json.dumps(second_map)),
            json.dumps(third_map),
            '{"name":"step150","districting":[',
        ]
    )
    if path.suffix == ".gz":
        with gzip.open(path, "wt", encoding="utf-8") as stream:
            stream.write(content)
    else:
        path.write_text(content, encoding="utf-8")
    return path


def test_read_atlas_records_skips_truncated_record(atlas_path):
    header, records, summary = read_atlas_records(atlas_path)

    assert header.magic == "This is an Atlas for Redistricting Maps"
    assert header.description == "Synthetic multiscale chain"
    assert header.atlas_param == {"districts": 2, "levels in graph": ["county", "prec_id"]}
    assert summary.chain_id == "toy-chain"
    assert summary.header_lines == 3
    assert summary.records_seen == 4
    assert summary.records_yielded == 3
    assert summary.bad_json_lines == 1
    assert summary.errors[0].startswith("line 7: bad JSON")

    assert [record.step for record in records] == [0, 50, 100]
    assert [record.line_number for record in records] == [4, 5, 6]
    assert records[0].weight == 2
    assert records[0].data == {"get_isoperimetric_scores": [10, 20]}
    assert records[0].assignment == {("North",): 1, ("South", "001"): 2}
    assert records[1].assignment == {("North",): 2, ("South", "001"): 1}


def test_read_atlas_records_thins_map_lines(atlas_path):
    _, records, summary = read_atlas_records(atlas_path, thin_step=2)

    assert [record.step for record in records] == [0, 100]
    assert [record.map_index for record in records] == [0, 1]
    assert summary.skipped_by_thinning == 2
    assert summary.bad_json_lines == 0


def test_read_atlas_records_strict_mode_rejects_truncated_record(atlas_path):
    with pytest.raises(AtlasRecordError, match=r"bad JSON .* at line 7:"):
        read_atlas_records(atlas_path, skip_bad_lines=False)
