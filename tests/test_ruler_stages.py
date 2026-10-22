"""OpenCompass timestamps must not break phase auditing or permit changed inputs."""
import json
import sys

import pytest

from mark_ruler_phase import main


def test_timestamp_layout_and_changed_predictions(tmp_path, monkeypatch):
    case = tmp_path / "case"
    for name in ("hf_runtime", "logs", "markers"):
        (case / name).mkdir(parents=True)
    for name in ("config.json", "runtime_audit.json"):
        (case / "hf_runtime" / name).write_text("{}")
    for phase in ("infer", "score"):
        (case / "logs" / f"{phase}.log").write_text("completed\n")
    for kind in ("predictions", "results"):
        directory = case / "opencompass/20261022_000000" / kind / "model"
        directory.mkdir(parents=True)
        for i in range(13):
            (directory / f"task_{i}.json").write_text("{}")
    monkeypatch.setattr(sys, "argv", ["mark_ruler_phase.py", "infer", str(case)])
    main()
    marker = json.loads((case / "markers/infer.json").read_text())
    assert len(marker["task_artifacts"]) == 13
    changed = case / marker["task_artifacts"][0]["path"]
    changed.write_text('{"changed": true}')
    monkeypatch.setattr(sys, "argv", ["mark_ruler_phase.py", "score", str(case)])
    with pytest.raises(ValueError, match="Inference outputs changed"):
        main()
    changed.write_text("{}")
    main()
    assert (case / "markers/score.json").is_file()
