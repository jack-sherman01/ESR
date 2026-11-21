# ESR 安装与统一运行入口

从源码 checkout 使用 `python -m esrb`；editable 安装后也可使用同一实现的 `esrb` 命令。
`experiments/` 保存实验流程，`esrb/` 保存公共能力。旧 Python 运行文件和兼容入口已删除。

## 1. 安装与环境

部署脚本面向 Linux x86_64 / Python 3.12：

```bash
bash scripts/setup_esr.sh install
bash scripts/setup_esr.sh smoke
bash scripts/setup_aspire.sh install
bash scripts/setup_aspire.sh smoke
```

已有环境更新 editable 安装：

```bash
uv pip install --python .venv/bin/python -e '.[sim]'
.venv/bin/python -m esrb --help
```

主项目 `.venv` 执行协调、研究实验和工具。ASPIRE 的 `.venv-robosuite` 执行原生机械臂仿真，`.venv-libero` 用于 LIBERO/感知诊断，各 provider 保留独立环境。
统一 CLI 会为原生 episode 和诊断选择相应解释器；ASPIRE 环境无需额外安装 ESR 包，子进程从当前源码 checkout 启动。

安装依赖 `configs/deployment/esr-constraints.txt`；本轮仅支持源码 checkout 和 editable 安装。
从其他目录调用已安装的 `esrb` 时，项目资源从包的位置定位，用户提供的相对输出路径按调用目录解释。
可用 `ESR_PROJECT_ROOT` 指向包含 `pyproject.toml` 和 `configs/` 的源码 checkout。

## 2. 命令分组

```text
esrb benchmark run                         综合自建任务实验
esrb benchmark study <name>                单场景、消融、多 seed 等研究
esrb benchmark search                      固定面板候选搜索
esrb aspire <task> chain <action>           单轮 50 版
esrb aspire <task> batch <action>           多轮，每轮最多 5 版
esrb tools <category> <name>                分析、报告、诊断、渲染、手动服务
```

五个正式场景名称：

| CLI 名称 | 原生 ASPIRE suite | Robosuite 环境 |
|---|---|---|
| `stack` | `cube_stack` | Stack |
| `pickplacecan` | `pickup_can` | PickPlaceCan |
| `two-arm-handover` | `two_arm_handover` | TwoArmHandover |
| `nut-assembly` | `nut_assembly` | NutAssemblySquare |
| `wipe` | `spill_wipe` | Wipe |

通过逐级 `--help` 查看参数，不会导入 MuJoCo、Torch 或 ASPIRE：

```bash
.venv/bin/python -m esrb benchmark study --help
.venv/bin/python -m esrb aspire pickplacecan chain init --help
.venv/bin/python -m esrb tools --help
```

## 3. 自建研究实验

```bash
MUJOCO_GL=osmesa .venv/bin/python -m esrb benchmark run \
  --config configs/experiment/default.yaml --scenarios tool_misuse \
  --generations 1 --rollouts 3 --out-dir NEW_OUTPUT

.venv/bin/python -m esrb tools analyze gate-audit NEW_OUTPUT/tool_misuse

.venv/bin/python -m esrb benchmark study ablation --help
.venv/bin/python -m esrb benchmark search --help
```

`studies` 保留 16 个研究流程，名称使用连字符，例如 `peg-insertion`、`episode-removal`、`sensor-drift`、`multiseed`。
MLP 任务需要可选 Torch：`bash scripts/setup_esr.sh ml`。受控更新、自主技能更新、参考评价的口径保持各自协议。

## 4. 显式准备服务

当前五场景技能 API 使用公开 RGB-D、PyRoki 和关节插值，不要求启动 SAM3 或 GraspNet。
先检查需要的端口；确认兼容的服务已运行时可以复用。

```bash
curl -fsS --max-time 5 http://127.0.0.1:8116/healthz

CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu \
.venv/bin/python -m esrb tools service pyroki -- \
  --urdf .esr-local/robot_descriptions/panda-d0d9098.urdf --port 8116
```

服务在前台运行，使用独立终端或计算作业管理。实验入口不自动启动服务，不自动创建 Agent，也不自动终止共享服务。

新长链的默认 IK 端口 / GPU 编号：Stack `8128 / 3`，PickPlaceCan `8116 / 0`，TwoArmHandover `8130 / 2`，NutAssemblySquare `8132 / 1`，Wipe `8131 / 0`。
按服务器资源准备对应服务，或初始化时显式覆盖：

```text
--ik-service-url http://127.0.0.1:8116 --gpu-device-id 0
--python /absolute/path/to/simulator/python
```

这些设置进入冻结协议，不能在执行时临时覆盖。多轮批次默认按轮分配 GPU；显式指定 GPU 后全部轮使用指定设备。

如需完整感知服务，按上游资产授权要求准备模型，再手动执行：

```bash
bash scripts/setup_aspire.sh assets
bash scripts/setup_aspire.sh serve
bash scripts/setup_aspire.sh check
```

令牌不要写入仓库或命令行。当前 checkout 路径由 `bash scripts/setup_aspire.sh path` 返回。

## 5. PickPlaceCan：固定 seed 连续 50 版

在源码根目录执行，使用全新输出目录和 worker 目录：

```bash
RUN="$PWD/.esr-local/pickplacecan/NEW_RUN"
CAPSULE="/tmp/pickplacecan-NEW_WORKER"

.venv/bin/python -m esrb aspire pickplacecan chain init \
  --root "$RUN" --capsule "$CAPSULE" --seed 42 \
  --ik-service-url http://127.0.0.1:8116 --gpu-device-id 0
```

省略 `--seed` 时随机选择一个，此后本轮全部版本使用该 seed，并检查实际模型和初态标识。
初始化从 `configs/experiment/` 冻结规则、从 `integrations/aspire/` 复制公开 API，并保存实际实现源码和哈希。

外部协调端显式创建一个无初始上下文的 worker，将简短任务和 `$CAPSULE/API.txt` 交给它。
当前 CLI 只接收文件提交，本身不调用模型；worker 的无上下文约束仍由外部协调端实施。
V1 从零编写，后续只使用本轮任务反馈和提供的起始程序，提示词不含宿主门控信息。

协调端写入 `$RUN/round_01/agent_assignment.json`：

```json
{
  "agent_id": "实际 worker ID",
  "capsule": "/tmp/pickplacecan-NEW_WORKER"
}
```

Worker 先写 `v01_attempt01.py`，完成后写 `ready_v01_attempt01.json`：

```json
{"rationale": "Improve grasping and placement."}
```

ready 后的提交不可修改。重复 AST 在仿真前拦截，worker 使用下一 attempt 重提同一版本，不占执行预算。

启动协调器：

```bash
.venv/bin/python -m esrb aspire pickplacecan chain drive --root "$RUN"
```

程序记录 `driver.log` 和 `driver_exit.json`。每版结束后产生 `feedback_vNN.json`、`additional_feedback_vNN.json` 和下一版 `starting_vNN.py`；worker 据此继续提交。
已有合格父策略时从父策略继续；尚无合格父策略时从上一版继续。成功后仍测满 50 版，不做独立复测。

只导出当前已有结果：

```bash
.venv/bin/python -m esrb aspire pickplacecan chain summarize --root "$RUN"
.venv/bin/python -m esrb tools report gate "$RUN"
.venv/bin/python -m esrb tools report chain --root "$RUN"
```

## 6. 单次技能和多轮批次

用示例技能执行一次带采集和门控的 PickPlaceCan：

```bash
.venv/bin/python -m esrb aspire pickplacecan batch init \
  --root NEW_BATCH --seeds 42 --versions-per-round 1

.venv/bin/python -m esrb aspire pickplacecan batch prepare \
  --root NEW_BATCH --round 1 --version 1 \
  --program examples/aspire/pickup_can_rgbd.py \
  --generation-base examples/aspire/pickup_can_rgbd.py \
  --rationale 'Initial task skill.'

.venv/bin/python -m esrb aspire pickplacecan batch episode \
  --root NEW_BATCH --round 1 --version 1
```

`init` 不指定 `--seeds` 时抽取 10 个不同 seed，轮内固定；每轮最多 5 版，成功不提前结束。
`wave --version N --workers K` 并行执行已准备的不同轮，轮内版本仍顺序执行。
`batch drive` 接收现有短批次提交协议：每轮 `agent_assignment.json` 包含 `agent_id`、`capsule`、`task_prompt`，capsule 由外部协调端准备 `API.txt`，worker 按 `vNN.py`、`ready_vNN.json` 提交。
它与长链的 `vNN_attemptMM` 协议不同，不能混用。

## 7. 诊断、证据和历史数据

不调用 Agent、不依赖 IK 的结构性采集 smoke：

```bash
MUJOCO_GL=osmesa .venv/bin/python -m esrb tools diagnose robosuite \
  --task pickplacecan --seed 42 --steps 3 --out NEW_SMOKE
```

该命令自动选择 ASPIRE 仿真解释器，独立 reset 两次，对照实际初态，记录短程关节保持动作、几何身份和物理采样。它不判断技能是否完成任务。
Nut 专用完整证据程序及审计通过 `tools diagnose nut-evidence --help` 访问；物理审计、指标重算和 GIF 渲染见 `tools analyze/render --help`。

每个正式版本保留程序、提案、执行结果、模型、初态、几何组、物理证据及宿主决定。缺测不能晋升；Wipe 保留其原生成功终止的特殊处理。

历史实验数据仍在 `.esr-local/{stack,pickplacecan,handover,nut_assembly,wipe}/`，各场景 `summary.json` 提供入口。
旧工作区 pilot、旧文件名命令和批次专用恢复工具已退役。严格重现历史批次需使用其冻结源码和原环境；新 CLI 不直接续跑旧协议，也不改写它们。

### 本机数据存储（2026-11-20 UTC）

实验数据已迁往 `/mnt/data1/zhifeng/ESR-data/`，保留项目内原路径的软链接：

| 项目入口 | 实际存放位置 |
|---|---|
| `.esr-local/{stack,pickplacecan,handover,nut_assembly,wipe}/` | `/mnt/data1/zhifeng/ESR-data/.esr-local/` 下各同名目录 |
| `.esr-local/{measurement-validation,measurement-development,_checks}/` | `/mnt/data1/zhifeng/ESR-data/.esr-local/` 下各同名目录 |
| `.esr-local/{_archives,_sources,esr}/` 及三个旧脚本归档目录 | `/mnt/data1/zhifeng/ESR-data/.esr-local/` 下各同名目录 |
| `private/analysis/` | `/mnt/data1/zhifeng/ESR-data/private/analysis/` |

本次共迁移 15 个目录。已有自然更新原始数据继续位于 `/mnt/data1/zhifeng/ESR-natural/20261118-r3-v50/`，没有重复复制；代码、展示素材、ASPIRE checkout 和 Python 环境保留在项目中。原路径可继续使用，但依赖 `/mnt/data1` 已挂载；通过软链接新增的文件也会写入数据盘。

迁移逐文件核对 SHA-256，并保留硬链接关系、权限和嵌套软链接；确认目标一致、原路径可访问后才移除源副本。清单和迁移回执位于 `/mnt/data1/zhifeng/ESR-data/storage-migration-20261120T043518Z/`；最终复核见 [verification.json](../.esr-local/_storage-migrations/20261120T043518Z/verification.json)。本次释放约 59.8 GiB，完成时 `/home` 可用约 64.8 GiB。迁移没有新增物理执行或修改实验标签。

## Panda 三件装配与输送线演示

使用项目 `.venv` 的 MuJoCo 4.0；不启动 ASPIRE 服务或调用外部模型。当前交付每场景一条 seed 0 成功轨迹，生成 5 秒 GIF，不追加多 seed 测试。

```bash
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m esrb \
  tools render physical-extensions \
  --acceptance .esr-local/_checks/panda-production-20261118/delivery/verification.json \
  --out-dir assets --width 640 --height 480 --duration 5
```

此命令只从原始轨迹渲染。`tools diagnose production-workcells` 仍供以后显式开展完整工程面板；其输出目录必须全新，不续跑旧冻结批次。完整模型、规则和说明见 [PANDA_PRODUCTION_WORKCELLS.md](PANDA_PRODUCTION_WORKCELLS.md)。

## Panda 接触工作台示例

```bash
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m esrb tools render physical-extensions --success-only \
  --scenarios constrained_push force_alignment compliant_assembly --seed 0 \
  --width 640 --height 480 --duration 5 --out-dir assets
```

每个场景运行一条轨迹，成功后生成 480p GIF。场景与开源参考见 [PANDA_CONTACT_WORKCELLS.md](PANDA_CONTACT_WORKCELLS.md)。
