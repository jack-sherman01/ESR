"""Fetch the exact authorized SAM3 revision and write a server-local receipt."""
import hashlib
import json
from pathlib import Path
import sys

from huggingface_hub import hf_hub_download

receipt = Path(sys.argv[1])
lock = json.loads(Path(__file__).with_name("sam3-source.json").read_text())
if receipt.exists():
    info = json.loads(receipt.read_text())
    if info["repo_id"] != lock["repo_id"] or info["revision"] != lock["revision"]:
        raise SystemExit("Existing receipt refers to a different model revision.")
else:
    info = {key: lock[key] for key in ("repo_id", "revision")}
    info["files"] = {}
    for name in lock["files"]:
        try:
            path = hf_hub_download(lock["repo_id"], name, revision=lock["revision"])
        except Exception as error:
            raise SystemExit(f"SAM3 download failed ({type(error).__name__}); check server network and hf auth login.") from None
        info["files"][name] = {"path": str(Path(path).absolute())}
for name, expected in lock["files"].items():
    entry = info["files"][name]
    path = Path(entry["path"])
    if path.stat().st_size != expected["bytes"]:
        raise SystemExit(f"SAM3 size mismatch: {name}")
    with path.open("rb") as handle:
        sha = hashlib.file_digest(handle, "sha256").hexdigest()
    if sha != expected["sha256"]:
        raise SystemExit(f"SAM3 hash mismatch: {name}")
    entry.update(expected)
if not receipt.exists():
    receipt.parent.mkdir(parents=True, exist_ok=True)
    with receipt.open("x") as handle:
        json.dump(info, handle, indent=2)
        handle.write("\n")
print(f"SAM3 assets verified: {receipt}")
