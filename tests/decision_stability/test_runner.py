import json
import subprocess
import sys

from experiments.decision_stability.model import load_model, smoke_config
from test_metrics_data import prepare_fixture
import torch


def test_checkpoint_data_cli_and_compute_accounting(tmp_path):
    prepare_fixture(tmp_path)
    config = smoke_config()
    config_path = tmp_path / "model.json"
    config_path.write_text(json.dumps(config))
    checkpoint = tmp_path / "weights.pt"
    torch.save(load_model(config).model.state_dict(), checkpoint)
    out = tmp_path / "run"
    cmd = [sys.executable, "-m", "experiments.decision_stability.run", "--config", str(config_path),
           "--checkpoint", str(checkpoint), "--data", str(tmp_path), "--limit", "2", "--out", str(out),
           "--steps", "3", "--fork-steps", "1", "--horizon", "3", "--probe-horizon", "2", "--targeted", "--epsilon", "0"]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["checkpoint_sha256"]
    assert manifest["compute"] == dict(natural_forward_sample_macros=6, continuation_forward_sample_macros=30,
                                        probe_forward_sample_macros=4, probe_backward_batch_calls=1,
                                        probe_backward_sample_objectives=2)
    records = [json.loads(line) for line in (out / "interventions.jsonl").read_text().splitlines()]
    clean = {(r["sample_id"], r["offset"]): r for r in records if r["arm"] == "clean"}
    for row in records:
        assert row["prediction"] == clean[row["sample_id"], row["offset"]]["prediction"]
        assert row["immediate_logits_equal"]
    result = subprocess.run([sys.executable, "-m", "experiments.decision_stability.analyze", str(out)],
                            check=True, capture_output=True, text=True)
    assert json.loads(result.stdout) == json.loads((out / "summary.json").read_text())
    assert subprocess.run(cmd, capture_output=True).returncode != 0  # refuses overwrite


def test_missing_checkpoint_never_silently_runs_random_model(tmp_path):
    result = subprocess.run([sys.executable, "-m", "experiments.decision_stability.run", "--out", str(tmp_path / "run")],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert "Research runs require" in result.stderr
