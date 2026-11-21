# Panda 接触工作台

`constrained_push`、`force_alignment`、`compliant_assembly` 已更换为完整七关节 Panda 和新的工业工装，模型版本为 `panda-contact/2/<scenario>`。三个名称保持不变，旧二维推杆和单件插入几何不再用于当前任务。

## 开源参考与本项目实现

| 当前场景 | 参考任务 | ESR 操作流程 |
|---|---|---|
| Constrained Push | [ManiSkill PushCube](https://github.com/mani-skill/ManiSkill/blob/main/mani_skill/envs/tasks/tabletop/push_cube.py) | 闭合夹爪接近托盘后沿导轨推送，经过收窄导入口进入定位区，撤离后验证托盘稳定 |
| Force Alignment | [ManiSkill AssemblingKits](https://github.com/mani-skill/ManiSkill/blob/main/mani_skill/envs/tasks/tabletop/assembling_kits.py) | 抓取初始有转角的工件，搬运并调整姿态，在导入口下降；接触导向面后消除横向偏移，沿导轨推进到目标位，松爪撤离 |
| Compliant Assembly | [ManiSkill PegInsertionSide](https://github.com/mani-skill/ManiSkill/blob/main/mani_skill/envs/tasks/tabletop/peg_insertion_side.py) | 从料架抓起带握持部的方轴，对准水平通孔，基于接触反馈推进，松爪后由孔壁与承托座支承 |

参考上游的任务组织、接触操作和配合判定思路，工装及显示网格为 ESR 原创，没有移植 ManiSkill 代码、模型或运行依赖。AssemblingKits 提供形状与目标槽配合的参考，接触触发纠偏是本项目扩展。来源记录见 `esrb/tasks/contact_workcells/assets/REFERENCES.json`。

机械臂复用现有 [MuJoCo Menagerie Panda](https://github.com/google-deepmind/mujoco_menagerie/tree/fb36886b3023db08c198264e1e9956ab202c3e3b/franka_emika_panda)，保留既有固定版本、网格、惯量、夹爪与许可证。零件使用独立的倒角显示网格和实体碰撞几何，工装包含导轨、导入口、定位面、承托结构与螺栓。

## 物理与控制

动作包含七个关节目标及夹爪闭合量。末端目标通过共享 Jacobian IK 转为关节指令，由真实执行器驱动完整 Panda。物理步为 2 ms、控制步为 20 ms，每回合最多 3500 控制步。所有工件均为自由刚体，通过重力、夹指摩擦及实体接触运动；阶段之间连续执行。唯一 equality 为 Panda 原生双指耦合。

演示使用 `simulator_state` 观测。对齐与插装根据实际零件姿态补偿抓持中的姿态变化，并从求解器接触记录计算导向面的反作用力。松爪时保持实际末端位姿，避免切换目标扰动已就位的工件。

完成条件检查实际目标位置、方位、倾斜及释放后的稳定状态，连续保持至少 0.5 秒；对齐任务还要求观察到真实导向面接触。位置工程容差为横向 4 mm、高度 5 mm；角度容差为 0.08 rad，托盘推送允许 0.12 rad 方位误差。稳定线速度低于 0.01 m/s，角速度低于 0.08 rad/s。这里提供工程示例，不是实机精度标定。

接触证据覆盖机器人、工件和设施；记录部件名称、接触点与力，区分推送、抓持、导向接触、支承和其他机器人接触。载荷与卡滞持续时间保留连续测量，旧模型的 20 / 35 N 等载荷限值未移用到新模型。新模型使用独立的资源身份、规则和策略 schema，不直接续跑旧冻结批次。

## 当前演示

每个场景使用 seed 0 保存一条成功轨迹；未开展多 seed 验收或 Agent 更新批次。对应 GIF 均为 **640×480、5 秒、单一近景、无放大窗口**：

- [Constrained Push](../assets/constrained_push.gif)：18.86 秒仿真，943 控制步。
- [Force Alignment](../assets/force_alignment.gif)：28.36 秒仿真，1418 控制步。
- [Compliant Assembly](../assets/compliant_assembly.gif)：21.26 秒仿真，1063 控制步。

原始轨迹、源码快照及来源哈希由各场景 `_animation.json` 指向；总交付记录在 `.esr-local/_checks/panda-contact-20261119/delivery/verification.json`。本次展示没有重复独立安全审计，元数据明确为 `NOT_RUN`，任务完成与安全资格分别记录。

通过统一 CLI 生成新的成功示例：

```bash
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m esrb tools render physical-extensions --success-only \
  --scenarios constrained_push force_alignment compliant_assembly --seed 0 \
  --width 640 --height 480 --duration 5 --out-dir assets
```

此命令每个场景只运行一个回合，成功后渲染该回合；失败会保留原始记录并报告原因。重新渲染已有交付轨迹：

```bash
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m esrb tools render physical-extensions \
  --acceptance .esr-local/_checks/panda-contact-20261119/delivery/verification.json \
  --scenarios constrained_push force_alignment compliant_assembly \
  --width 640 --height 480 --duration 5 --out-dir assets
```

修改前源码、旧素材及哈希清单保存在 `.esr-local/_archives/panda-contact-20261119-005822/`。调试失败轨迹保留在 `.esr-local/_checks/panda-contact-20261119/`，不作为成功演示或成功率统计。
