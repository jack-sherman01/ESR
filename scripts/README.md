# 脚本目录

脚本按功能分组；核心任务和判定实现位于 [`esrb/`](../esrb/)，ASPIRE 服务及公开 API 说明位于 [`integrations/aspire/`](../integrations/aspire/)。

| 子目录 | 用途 |
| --- | --- |
| [runners/](runners/) | 场景批次、单轮 50 版更新链和执行协调 |
| [gates/](gates/) | 物理采集、规则加载、晋升记录及共享状态采集函数 |
| [feedback/](feedback/) | 生成 Agent 可见的任务反馈及 API 错误过滤 |
| [adapters/](adapters/) | Wipe 固定工具的运动接口兼容和坐标换算 |
| [reports/](reports/) | 从已有批次导出执行、任务和门控结果 |
| [analysis/](analysis/) | 固定面板分析、物理证据审计和装配重放验证 |
| [recovery/](recovery/) | 保留已有记录并恢复中断的更新链 |
| [render/](render/) | README 图像、GIF 和受控任务动画 |
| [setup/](setup/) | ESR/ASPIRE 环境安装和服务检查 |

五场景单轮 50 版入口：

| 场景 | 脚本 |
| --- | --- |
| Stack | [run_stack_long_chain.py](runners/run_stack_long_chain.py) |
| PickPlaceCan | [run_pickup_can_long_chain.py](runners/run_pickup_can_long_chain.py) |
| TwoArmHandover | [run_handover_long_chain.py](runners/run_handover_long_chain.py) |
| NutAssemblySquare | [run_nut_long_chain.py](runners/run_nut_long_chain.py) |
| Wipe | [run_wipe_long_chain.py](runners/run_wipe_long_chain.py) |

查看参数和安装路径：

```bash
python scripts/runners/run_stack_long_chain.py --help
bash scripts/setup/setup_aspire.sh path
```

实际仿真使用 ASPIRE 专用解释器和所需服务。更新链还需要独立生成 Agent 按工作目录协议提交程序；启动驱动不会自动提供模型服务。完整步骤见 [SERVER_SETUP.md](../docs/SERVER_SETUP.md)，测试说明见 [tests/README.md](../tests/README.md)。

实验输出放在被 Git 忽略的 `.esr-local/<场景>/`，人工研究记录放在 `private/`。已有批次绑定当时的源码和规则哈希，严格复现应使用其源码快照。独立 Nut smoke 入口已移除；仍被运行链使用的基础函数保留在 [simulation_capture.py](gates/simulation_capture.py)。
