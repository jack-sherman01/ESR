# Deploy ESR and ASPIRE on a Linux server

These steps use the checked-out ESR repository. No copy of a laptop's virtual
environments or an existing `ESR_benchmark` project is required.

## 1. Install ESR

The scripted setup targets Linux x86_64 and Python 3.12. Install Git, curl,
Python 3.12, and [uv](https://docs.astral.sh/uv/getting-started/installation/).
Use your cluster's Python module if needed, or set `ESR_PYTHON=/path/to/python3.12`
before running these scripts.

```bash
git clone https://github.com/jack-sherman01/ESR.git
cd ESR
bash scripts/setup_esr.sh install
bash scripts/setup_esr.sh smoke
```

If you are reviewing an unmerged setup branch, clone that branch with
`git clone --branch feat/server-setup https://github.com/jack-sherman01/ESR.git`.

Installation creates `.venv/`. The smoke test runs one small `tool_misuse`
generation and audits its saved result. It disables rendering. No GPU or
model credentials are needed for this check. A rejected candidate can be the
expected result; check the audit's `PASS`. By default logs and outputs are
written under the Git-ignored `.esr-local/` directory.

For `mlp_peg` and `episode_removal`, install CPU PyTorch separately:

```bash
bash scripts/setup_esr.sh ml
```

## 2. Install ASPIRE

For NVIDIA GPU simulation, verify `nvidia-smi` and EGL are available on the
assigned compute node. The pinned upstream ASPIRE environments use PyTorch
with CUDA 12.8 for GPU providers. Check the [NVIDIA driver compatibility
guide](https://docs.nvidia.com/deploy/cuda-compatibility/latest/minor-version-compatibility.html)
and your cluster's EGL setup. Install dependencies on a node that can reach
GitHub and the package indexes. For a first single-GPU deployment, budget
about 60 GB of disk for environments, caches, and a few runs.

```bash
bash scripts/setup_aspire.sh install
bash scripts/setup_aspire.sh smoke
```

The install command fetches a pinned upstream [NVlabs/ASPIRE](https://github.com/NVlabs/ASPIRE)
checkout, applies the LIBERO saved-state fix, and invokes the upstream frozen
installers for LIBERO-Pro, Robosuite, SAM3, Contact-GraspNet and PyRoki.
`bash scripts/setup_aspire.sh path` shows where ASPIRE was installed.
By default this is `.esr-local/aspire/checkout`. Set an absolute
`ESR_ASPIRE_DIR` before every ASPIRE command to store the checkout elsewhere.
Use a new directory; the installer refuses a checkout at a different commit.
Run `bash scripts/setup_aspire.sh source` to fetch just the source.
The install uses the upstream `rtx4090` single-card profile by default;
set `CAP_HARNESS_PROFILE` to a profile appropriate for your GPU if needed.

ASPIRE's Python environments are separate from ESR's. The smoke test runs a LIBERO-Pro
reset and three control steps, then a Robosuite cube reset. It does not download
SAM3 weights or call a coding Agent.

## 3. Enable local ASPIRE model services

Log in on the server with an account that has [access to SAM3](https://huggingface.co/facebook/sam3).
Do not put your token in the Git repository or pass it on the command line.

```bash
ASPIRE="$(bash scripts/setup_aspire.sh path)"
export HF_HOME="$PWD/.esr-local/huggingface"
"$ASPIRE/.venv-sam3/bin/hf" auth login
bash scripts/setup_aspire.sh assets
"$ASPIRE/.venv-sam3/bin/hf" auth logout
```

`assets` downloads and verifies the pinned SAM3 checkpoint (about 3.45 GB),
verifies the GraspNet checkpoint from its pinned submodule, and prepares the
Panda URDF cache. Keep the downloaded model cache after logging out. If you
see a Panda cache revision conflict, use a separate deployment account or ask
your administrator to arrange a separate cache. The installer does not change an
existing mismatched cache.

In one terminal, start the services. Use a foreground session or a compute
job, and stop it with Ctrl+C. Use a dedicated or device-isolated GPU on
multi-user systems so the EGL renderer and SAM3 use the assigned GPU.

```bash
bash scripts/setup_aspire.sh serve
```

In another terminal from the same ESR checkout:

```bash
bash scripts/setup_aspire.sh check
```

The ports are local only: `8114` SAM3, `8115` Contact-GraspNet,
and `8116` PyRoki. This recipe runs SAM3 on a GPU with weight offload
by default and GraspNet/PyRoki on CPU. The check includes an actual SAM3
perception request using a simulated LIBERO image, plus health checks for all
three services. Deployment commands do not require a coding Agent account.

This setup covers the LIBERO-Pro/Robosuite runtime and the three services
listed above. cuRobo, BEHAVIOR-1K, physical hardware, and private ESR+ASPIRE
research policies/runs are outside this public install recipe. To run ASPIRE
experiments requiring other providers or an Agent, follow the docs in the
pinned ASPIRE checkout.
