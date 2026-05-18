import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]


def test_single_card_multitile_ccl_semantics_probe_passes():
    script = REPO_ROOT / "scripts/probe_single_card_multitile_ccl_semantics.py"

    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "participant_count=4" in result.stdout
    assert "all_gather_shape=[1, 2, 256, 1024]" in result.stdout
    assert "single_card_multitile_ccl_semantics=ok" in result.stdout
    assert "all_gather_ok=true" in result.stdout
    assert "reduce_scatter_ok=true" in result.stdout
    assert "all_to_all_ok=true" in result.stdout
