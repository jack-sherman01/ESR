"""Logged external process calls for explicit model-driven searches."""
import json
import os
import subprocess
from .io import read, save_new as save

def run_logged(command, cwd, output, *, env, timeout, stdin=None, stdout="stdout.log"):
    output.mkdir(exist_ok=True, parents=True)
    save(output / "invocation.json", {"command": command, "timeout_s": timeout})
    with (output / stdout).open("xb") as out, (output / "stderr.log").open("xb") as err:
        try:
            exit_code = subprocess.run(
                command, cwd=cwd, env=env, stdout=out, stderr=err, input=stdin,
                timeout=timeout, check=False,
            ).returncode
        except subprocess.TimeoutExpired:
            save(output / "status.json", {"status": "timeout"})
            raise
        except OSError as exc:
            save(output / "status.json", {"status": "launch_failed", "error": str(exc)})
            raise
    save(output / "status.json", {"exit_code": exit_code})
    if exit_code:
        raise RuntimeError(f"process exited {exit_code}; see {output}")

def child_env():
    return {
        key: value for key, value in os.environ.items()
        if key not in {"PYTHONPATH", "PYTHONHOME"}
        and not key.endswith(("_API_KEY", "_TOKEN"))
    } | {"PYTHONDONTWRITEBYTECODE": "1"}
