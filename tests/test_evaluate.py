import json

from zeppelin_rag.config import TESTSET_PATH
from zeppelin_rag.evaluate import METRIC_NAMES, load_testset, summarize


def test_testset_is_well_formed():
    items = load_testset(TESTSET_PATH)

    assert len(items) >= 15
    for item in items:
        assert item["question"].strip() and item["reference"].strip()


def test_summarize_skips_failed_metrics():
    rows = [
        {name: 1.0 for name in METRIC_NAMES},
        {**{name: 0.5 for name in METRIC_NAMES}, "faithfulness": None},
    ]

    summary = summarize(rows)

    assert summary["faithfulness"] == 1.0
    assert summary["context_recall"] == 0.75
    assert json.dumps(summary)
