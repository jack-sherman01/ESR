# Panda 工业工作台：三件装配与输送线

三个自建场景使用完整的七关节 Franka Panda、耦合双指夹爪和 MuJoCo 接触动力学。场景接口名称保持不变，测量模型为 `panda-production/2/<mode>`；这是本项目的工程任务扩展，不是 Robosuite / MimicGen 原生环境的复现成绩。

## 场景分工

| 场景 | 操作链 | 终态要求 |
|---|---|---|
| `long_horizon_assembly` | 从初始机械臂姿态出发，抓底座放入定位座，装连接件，再装上部件 | 三件同时满足配合条件，释放并撤离后稳定至少 0.5 秒 |
| `moving_transport` | 跟踪持续进料的三件工装，依次随动抓取、抬离护栏、定向放入配套料盘 | 三件均进入各自槽位并稳定释放；错过抓取窗口明确失败 |
| `moving_assembly` | 底座预置在自由托盘定位槽中；抓取两个后续部件，在托盘持续移动时完成装配 | 全部配合稳定，并且两次装配期间托盘的实际平均速度均超过带速的一半 |

每次操作经历 `approach → descend → close → lift → transfer → lower → release → retreat → verify_object`，最后共同验证整个组件。阶段由实际位姿、双侧夹持接触、抬升与释放状态推进。装配之后的步骤可以扰动前面的结果，阶段之间不重置状态。

底座与两件配合件均为自由刚体；移动托盘也是自由刚体。唯一 equality 是 Panda 原生双指耦合，没有抓取或装配焊接。机器人只能通过执行器运动，零件通过摩擦、重力与实体接触运动。基准使用实际零件姿态补偿侧握底座的重力翻转。

## 来源与模型

- [MuJoCo Menagerie Panda](https://github.com/google-deepmind/mujoco_menagerie/tree/main/franka_emika_panda)：沿用项目已固定的提交 `fb36886b3023db08c198264e1e9956ab202c3e3b`，保留原模型网格、惯量、关节限位与原生夹指几何。关节伺服参数和重力补偿沿用 ESR 工程配置，未声称经过实机标定。
- [MimicGen ThreePieceAssembly](https://github.com/NVlabs/mimicgen/blob/c0fbff22ba40060f05a917a80817f3f5c20731e7/mimicgen/envs/robosuite/three_piece_assembly.py)：固定提交 `c0fbff22ba40060f05a917a80817f3f5c20731e7`。移植三件占据格与单元尺寸，合并相邻实心格生成等价碰撞盒，保留孔槽。增加底座握持区、独立倒角显示网格、料盘和托盘工装。
- [MuJoCo 输送线抓取参考](https://github.com/esung52/Mujoco-UR5E-Robot-Picking)：借鉴运动跟踪与拦截抓取思路，没有引入其 UR5e 控制器或运行依赖。
- 输送带使用 [MuJoCo `surfacevel`](https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-geom-surfacevel)，通过接触切向摩擦输送物体；控制器读取物体的实际速度，包含带面与零件间的真实滑移。

工程件采用有限质量的轻量原型参数，底座、连接件、上部件质量约为 0.091、0.147、0.211 kg。显示颜色不代表实物材料标定。MimicGen 完整许可证、来源文件哈希与移植说明在 `esrb/tasks/production/assets/`，Panda 原始许可证和资源清单在其既有资源目录。

## 控制、初态与判定

物理步为 2 ms，控制步为 20 ms，每回合上限 6000 控制步 / 120 秒。动作是七个关节目标角及夹爪闭合量，末端路点通过共享的六维 Jacobian IK 转为有界关节指令。首轮观测明确标为 `simulator_state`，不依赖模型推理、视觉训练或实机部署。

名义搬运带速 0.04 m/s，装配带速 0.005 m/s；工程初态域包含整体位置扰动 ±1 mm、方位扰动 ±0.01 rad、摩擦系数 0.85–0.95 和带速倍率 0.95–1.05。静态场景不启用输送运动。这是有界工程验收域，未声称覆盖上游完整随机分布。

配合判定使用 4 mm 位置容差、0.08 rad 方位及倾斜容差、0.008 m/s 稳定相对速度、0.05 rad/s 相对角速度和 0.5 秒连续验证。配合误差在实际底座/托盘的完整三维坐标系内计算；稳定性扣除参考工装的平移和旋转速度，并将质心速度换算到零件原点。最终任务判定还要求夹爪确实打开并撤离，不能仅依靠程序走完。

任务成功和 ESR 安全结果分别保留。安全规则覆盖所有 Panda 连杆、夹指、零件和工装，分类记录抓持、装配、支承和非预期机械臂接触。掉落要求抬升后失去夹持、发生向下分离及真实下落；被槽边阻挡而失去接触不能直接判为掉落。掉落后额外保留 0.5 秒物理轨迹。

逐物理步保存状态、实际执行器指令/力矩、接触部件、位置、接触坐标与世界坐标力和带速。ESR 重新计算 FK、求解器接触、阶段连续性和最终配合；关键证据缺失或身份不匹配输出 `UNKNOWN`。活动零件接触载荷和卡滞时长先保留连续测量，没有移用原单件插入的 35 N / 1 Nm 限值。

受控候选分别减少就位等待、削弱搬运夹持与速度补偿、使用过时工装位姿。只修改策略动作，不注入掉落或碰撞。候选可能只有任务失败而没有安全违规，报告保留实际结果。

## 本次交付：成功轨迹与 GIF

根据最新要求，不再追加多 seed 或受控候选迭代。当前版本每个场景使用 seed 0 完成一次任务，保存完整实测轨迹，渲染一份 640×480、5 秒成功 GIF。任务完成由实际配合、释放与稳定状态确认；本次演示轨迹不重复独立安全审计，元数据明确记为 `NOT_RUN`，不据此宣称安全资格。

原始数据与源码快照：`.esr-local/_checks/panda-production-20261118/delivery/`。`verification.json` 记录三个任务的完成结果、模型身份及原始数据哈希。演示元数据记录实际仿真时长、播放倍率、初态、策略、模型和轨迹来源。GIF 为固定近景、无放大窗口；`assets/*_workcell.png` 另给完整 Panda 全景。

从已保存轨迹重新渲染，不增加仿真回合：

```bash
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m esrb tools render physical-extensions \
  --acceptance .esr-local/_checks/panda-production-20261118/delivery/verification.json \
  --out-dir assets --width 640 --height 480 --duration 5
```

此前工程调试数据继续保留。`acceptance-v2/verification.json` 的 33 个回合属于冻结的 `panda-production/1`，其中 30 个基准成功、3 个受控候选失败；它不是当前 `/2` 版本的多 seed 成绩。后续如需完整工程面板，可显式调用 `tools diagnose production-workcells`，默认会运行多 seed 和受控候选，本次不再执行。

修改前的源码及旧素材保存在 `.esr-local/_archives/panda-production-20261118-222530/`。`compliant_assembly` 已另行升级为 [Panda 水平插装工作台](PANDA_CONTACT_WORKCELLS.md)。历史三个旧场景的球体干扰、笛卡尔滑台和相关验收数值属于其冻结源码，不能与本版成绩混合比较。本次未启动正式 Agent 更新批次或实机部署。
