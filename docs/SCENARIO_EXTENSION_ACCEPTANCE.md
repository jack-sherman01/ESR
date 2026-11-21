> Current contact-model notice: `constrained_push`, `force_alignment`, and `compliant_assembly` now use full Panda workcells. The old pusher/insertion results below remain historical evidence. See [current contact tasks and demos](PANDA_CONTACT_WORKCELLS.md).

> Current-model notice: `moving_transport` and `moving_assembly` now use full Panda conveyor workcells. The moving-sphere results in this document are historical frozen evidence. See [current task and acceptance documentation](PANDA_PRODUCTION_WORKCELLS.md).

# 第三、四批仿真工程验收

范围：确定性受控工程验收；不是论文样本、自然 agent 更新或真机安全认证。

## 交付

- `moving_transport` / `moving_assembly`：有碰撞的运动代理在持续 episode 中穿越，参数化开始时间、直线路径、速度、半径和高度；等待、提前通过、违规穿越均有真实轨迹。每个物理步检查所有机器人几何与自由物体的最近表面距离，保护夹爪、外壳和所持物体，记录部件和发生时刻。
- `constrained_push`：自由刚体、物理通道和推杆。检测外力/力矩、推杆载荷、导轨载荷、末端挡块禁碰和完整三维物体轮廓越界。
- `force_alignment`：窄推杆通过偏心接触纠正自由零件位置/姿态，结合实测载荷退让和物理漏斗导向，终态要求真实对齐与稳定保持。
- `compliant_assembly`：插座几何实际偏移，接触导纳与已知夹具位姿基准联合修正；完整经历抓取、运输、预定位、插入、松爪、稳定验证，增加持续插入力与卡滞时长规则。
- 五者均接入 YAML、受限物理难度/扰动域、结构化 reset、统一提案、影响图测试、反例重放/缩减、独立风险抽样与离线审计。缺失或矛盾证据标为 UNKNOWN，禁止接纳；普通任务失败与安全违规分开记录。

## 检查

完整回归 **386 项通过**（505.52 s）。全套收集后追加的“原有装配不得通过日志伪造插座位置”专项 **1 项通过**；其后原有连续装配 **10 项复验通过**。新增场景测试包含正常完成、危险更新、缺测/伪造读回、任务失败、阶段/episode 超时、精确重放、参数拒绝、两代历史重放和版本追溯。还单独验证了运动代理可对机器人外壳施加真实接触，而外壳不会与所持物体发生错误自碰撞。

最终选用命令行面板：每个场景 1 代，父策略 8 次、候选总预算 8 次，其中独立风险 2 次、边界 1 次、缩减 4 次、fresh 至少 1 次。每次缩减均实际执行并记录；五个场景的 retained witness 都经执行确认。独立审计再次复算全部规则、来源、阶段、提案与预算，五项均为 PASS。

| 场景 | 父策略任务完成 | 候选明确违规 | 候选缺测 | 接纳代数 | 独立审计 |
|---|---:|---:|---:|---:|---|
| `moving_transport` | 8/8 | 8/8 | 0 | 0 | PASS |
| `moving_assembly` | 8/8 | 8/8 | 0 | 0 | PASS |
| `constrained_push` | 8/8 | 8/8 | 0 | 0 | PASS |
| `force_alignment` | 2/8 | 8/8 | 0 | 0 | PASS |
| `compliant_assembly` | 8/8 | 6/8 | 0 | 0 | PASS |

最终选用面板共 **40 次父策略 + 40 次候选 rollout**，父控制步 16153、候选控制步 18688，合计 **348410 个物理步**。此统计仅指五个选用报告，不包括单元测试及先前被替代的工程运行；不把它称为整个开发过程的总成本。另有两个运动场景的早期验收被最终外壳碰撞掩码版本替代，旧文件保留，未混入最终面板。

注意：`force_alignment` 在这组有限扰动父面板中只有 **2/8 任务完成**；它在标称参考工况可完成，但控制器还不够鲁棒，不能宣称整个扰动域都可稳定完成。柔顺装配的危险候选中有 2/8 未观察到明确安全违规，报告没有把所有候选强行标成危险。每组两条独立风险样本不足以证明 epsilon=0.09 的安全率；这些结果只验证机制。

## 复现与证据

```bash
MUJOCO_GL=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m esrb benchmark run \
  --config configs/experiment/physical_extensions.yaml --skip-sweep \
  --out-dir /tmp/esrb-physical
.venv/bin/python -m esrb tools analyze gate-audit /tmp/esrb-physical/moving_assembly
```

配置默认执行两代；加 `--generations 1 --fresh-min 1 --reduction-budget 4` 可复现本次单代面板的预算结构。请使用新的输出目录。

本机最终证据：

- `private/runs/physical_extensions_work_20261116/verified/acceptance.json`：五场景最终来源、文件哈希、规则复算、成本、软件版本和源文件摘要。
- `private/runs/moving_extensions_verified_20261116/`：最终两个运动场景原始报告和反例。
- `private/runs/physical_extensions_20261116/`：三个接触任务的原始报告和反例；其中两个旧运动场景报告仅供溯源。
- `private/runs/physical_extensions_work_20261116/visuals/`：从保存的物理状态直接渲染的父/候选对照图及用例来源，没有重新生成演示轨迹。

```bash
.venv/bin/python -m esrb tools analyze audit-physical-extensions \
  --run private/runs/physical_extensions_20261116 \
  --run private/runs/moving_extensions_verified_20261116 \
  --out /tmp/esrb-physical-re-audit
```

后一个 run 显式取代同名场景，旧来源及文件摘要仍写入汇总。此命令只离线重建测量/规则，不启动新的仿真 rollout。详细限值、坐标、阶段与成本范围见 [工程协议](SCENARIO_EXTENSION_PROTOCOL.md)。

未使用外部模型服务、未做真机动作、未采集论文研究样本；没有提交 Git。
