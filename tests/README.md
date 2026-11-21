# tests 目录说明

这里保存自动化测试代码，检查修改后是否破坏已有行为。它不保存正式仿真批次的数据；实验原始记录在 `.esr-local/` 和 `private/`。

测试模块按被测功能分到四个目录。配置夹具集中放在 `fixtures/`；各级 `__init__.py` 保留测试包导入。部分模块同时包含纯逻辑测试和 MuJoCo 集成测试，所以目录分类不代表运行成本或依赖等级。

```text
tests/
├── core/        核心框架、门控与评价
├── tasks/       自建任务、策略与更新算子
├── aspire/      ASPIRE 接口与场景门控
├── workflows/   Agent 反馈与长链恢复
├── fixtures/    测试使用的静态配置样本
├── __init__.py
└── README.md
```

| 目录 | 类别 | 模块数 | 作用 |
| --- | --- | ---: | --- |
| [core/](core/) | 核心框架、门控与评价 | 14 | 候选提案、影响分析、风险抽样、历史反例、父策略晋升、评价指标与配置校验 |
| [tasks/](tasks/) | 自建任务、策略与更新算子 | 15 | 插孔、抓取放置、工具、动态障碍、装配，以及 MLP、记忆污染、回合删除等受控更新 |
| [aspire/](aspire/) | ASPIRE 接口与场景门控 | 16 | Stack、PickPlaceCan、TwoArmHandover、NutAssemblySquare、Wipe 的反馈、接触身份、采集和判定 |
| [workflows/](workflows/) | Agent 反馈与长链恢复 | 9 | 提示和反馈隔离、程序去重、长链协议、中断归档、恢复和子进程退出码 |

新增测试按被测功能放入对应目录：项目通用机制放 `core/`，自建仿真任务与策略放 `tasks/`，ASPIRE 场景采集和判定放 `aspire/`，跨回合 Agent 执行与恢复放 `workflows/`。

## core/：核心框架、门控与评价

- `test_candidate_proposals.py`：提案格式、宿主消费接口、无效提案在仿真前拒绝。
- `test_detection.py`、`test_gate_planner.py`：改动影响、检查选择和历史反例排程。
- `test_risk_sampling.py`、`test_multiconstraint.py`：风险采样预算、多约束证据和审计。
- `test_oracles.py`、`test_proximity_oracle.py`：力、力矩、禁区和障碍距离规则。
- `test_fixed_chain.py`、`test_cube_generation_gate.py`：不同实验协议各自冻结的晋升规则。
- `test_evaluation_v2.py`、`test_metrics.py`：v2 参考评价和保留的 v1 指标接口，不能互相替代。
- `test_experiment_config.py`、`test_structured_resets.py`：配置合法性、旧配置兼容和初态重放。
- `test_evolution_loop.py`：完整更新循环、谱系和反例库。

## tasks/：自建任务、策略与更新算子

- `test_peg_insertion.py`、`test_peg_gate_cases.py`：插孔环境、边界初态、重放和反例缩减。
- `test_pick_place.py`、`test_dynamic_obstacle.py`：搬运和动态避障。
- `test_tool_handling.py`、`test_tool_misuse.py`、`test_tool_tighten.py`：工具抓持、误用与拧紧。
- `test_long_horizon_assembly.py`、`test_production_workcells.py`：完整 Panda、三件装配、真实输送、连续阶段、释放验证及任务失败/安全违规分离。
- `test_physical_extensions.py`：完整 Panda 推送、对齐与水平插装的八维动作、物理初态、证据缺测、策略消费者及有界超时；旧推杆专用测试已退役。
- `test_mlp_policy.py`、`test_episode_removal.py`：策略训练、记忆污染、缓冲区删除及其回归。
- `test_evaluator_drift.py`、`test_sensor_drift.py`、`test_stochastic_force.py`：评价器、感知和随机载荷变化。

这些任务仍由当前实现和实验入口提供。它们与最近的上游机械臂任务用途不同，不能仅因运行时间较早而删除。

## aspire/：ASPIRE 接口与场景门控

- `test_stack_batch.py`、`test_stack_gate.py`、`test_stack_gate_report.py`、`test_stack_task_feedback.py`：Stack 批次、门控、导出和反馈。
- `test_pick_place_can.py`、`test_pickup_can_gate.py`：易拉罐稳定放置与安全门控。
- `test_handover_gate.py`：交接时机、接收抓持、掉落、双臂接触和反馈隔离。
- `test_simulation_capture.py`、`test_nut_episode.py`、`test_nut_online_gate.py`：接触部件身份、螺母就位、掉落和在线门控。
- `test_wipe_gate.py`、`test_wipe_gate_state.py`、`test_wipe_capture.py`：擦拭门控状态、逐物理步采集和缺测处理。
- `test_wipe_mapping.py`、`test_wipe_public_api.py`：固定擦拭工具的坐标换算和公开运动接口。

## workflows/：Agent 反馈与长链恢复

- `test_fresh_agent_feedback.py`：保留公开任务/API 错误，排除宿主门控信息。
- `test_stack_long_chain.py`、`test_pickup_can_long_chain.py`、`test_other_long_chains.py`：50 版协议、不可变提案和去重。
- `test_stack_long_chain_recovery.py`：公共中断归档保护，禁止覆盖已测结果；历史批次专用恢复脚本已退役。
- `test_long_chain_supervisor.py`：记录真实子进程退出码。
- `test_script_layout.py`：统一 CLI 全部注册命令的帮助、轻量导入、跨目录启动和安装脚本定位。
- `test_common_runtime.py`：五场景公共引擎、每场景50版模拟提交、去重和父策略保留，以及迁移前判定结果对照。

工作流测试直接调用正式公共模块，不通过旧入口或兼容层。

## 配置夹具

[`fixtures/legacy_peg_profile.yaml`](fixtures/legacy_peg_profile.yaml) 是 [`core/test_experiment_config.py`](core/test_experiment_config.py) 使用的旧格式配置样本，用于验证旧配置仍可加载、非法限值会拒绝，以及原配置文件删除后仍能审计归档。`legacy` 在这里表示兼容性样本，当前仍有测试引用。

## 运行与可选依赖

在已安装项目依赖和 pytest 的项目环境中：

```bash
MUJOCO_GL=osmesa .venv/bin/python -m pytest tests -q
```

可以按目录运行，例如只检查场景接口和工作流：

```bash
.venv/bin/python -m pytest tests/aspire tests/workflows -q
```

也可以只检查某个场景及其恢复流程：

```bash
.venv/bin/python -m pytest tests/aspire/test_stack_gate.py tests/workflows/test_stack_long_chain_recovery.py -q
```

- MuJoCo 集成测试需要 MuJoCo；部分模块在缺少它时跳过。
- MLP 相关测试另外需要可选 PyTorch。缺少它时，只跳过依赖 MLP 的用例。
- `test_wipe_public_api.py` 需要可选 ASPIRE 的 `cap_harness`；普通 ESRB 环境会跳过该模块。配置 ASPIRE 后应单独验证它：

```bash
.esr-local/aspire/checkout/.venv-robosuite/bin/python -m pytest tests/aspire/test_wipe_public_api.py -q
```

## 本次清理

- 删除 `test_episode_removal.py` 中与 `test_mlp_policy.py` 相同训练参数、初态和断言的干净策略基线测试；保留单一覆盖位置。
- 删除 `test_retrain_shifts_output_toward_contaminated_target`：它只断言绝对值非负，没有检验所声称的学习方向。已有显式重训练运行测试和污染后物理违规集成测试保留。
- 清理不需要的 Python 字节码缓存。
- 修正缺少可选 PyTorch/ASPIRE 时的跳过条件，避免阻断其余测试。
- 保留仍有对应实现的测试模块及旧配置兼容夹具。
- 将 52 个测试模块归入四个功能目录，修正脚本、配置及夹具路径，并同步文档中的测试命令。`pytest tests` 和现有 CI 会递归发现这些目录。

## 统一入口重构

实际实现位于 `experiments/`、`esrb/runtime/` 和 `esrb/backends/aspire/`。删除旧工作区 pilot 与批次专用恢复测试，保留公共输入过滤和中断证据保护。`fixtures/admission_decisions.json` 保存迁移前七类输入在五场景下的判定，检验 Wipe 原生终止等语义未改变。
