"""Check the pinned ASPIRE tracked source delta before bootstrapping."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

root = Path(sys.argv[1])
lock = json.loads(Path(__file__).with_name("aspire-source.json").read_text())
git = ["git", "-C", str(root)]
assert subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip() == lock["upstream_commit"]
changed = subprocess.check_output(
    [*git, "diff", "--name-only", "--ignore-submodules=all", "HEAD", "--"], text=True
).splitlines()
# New files in the patch are still untracked until a local commit is made.
extra = set(changed) - lock["patched_files"].keys()
if extra:
    raise SystemExit(f"Unexpected edits in pinned ASPIRE source: {sorted(extra)}")
for rel, sha in lock["patched_files"].items():
    if hashlib.sha256((root / rel).read_bytes()).hexdigest() != sha:
        raise SystemExit(f"ASPIRE file does not match the migration snapshot: {rel}")
print("ASPIRE upstream commit and local patched files verified.")
