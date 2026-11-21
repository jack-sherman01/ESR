# 图像与实验预览

| 位置 | 内容 |
| --- | --- |
| 当前目录 | README 使用的自建任务插图、曲线、动画和渲染说明 |
| [aspire/](aspire/) | 官方 ASPIRE 工作流产生的预览 |
| [gpt sol 6.1/](<gpt sol 6.1/>) | 由初始无上下文 GPT Sol Agent 直接编写技能得到的预览和摘要 |

历史 ASPIRE 预览将整段已录制视频加速为 5 秒：

| 来源 | 场景 / 版本 / seed | 结果 |
| --- | --- | --- |
| 官方 ASPIRE | TwoArmHandover / V09 / 123 | **任务失败**，保留作失败示例 |
| GPT Sol | TwoArmHandover / V50 / 82478114 | 原生成功，门控晋升 |
| GPT Sol | Stack / V50 / 624537285 | 原生成功，门控晋升 |
| GPT Sol | PickPlaceCan / V50 / 262045286 | 原生成功，门控晋升 |

GIF 同名的 `.source.json` 保存来源、哈希和时长信息。`pickplacecan_single_round_50.summary.json` 汇总一个固定 seed 上 50 个不同程序的执行结果；它不是同一策略在 50 个独立 seed 上的成功率。

最新自然更新批次的五场景成功动画位于 [gpt sol 6.1/natural-seeds1to5/](<gpt sol 6.1/natural-seeds1to5/>)：
Stack、PickPlaceCan、TwoArmHandover、NutAssemblySquare 使用第一链 V50，Wipe 使用第二链 V50，均选 seed 1。
这些程序均在开发 seed 1–5 上任务成功。本批没有 ESR 接纳或安全判定。
动画按归档的原生动作重新以 640×480 渲染，逐控制步核对原始状态，整回合加速为5秒；
Wipe 保留原生污点擦除过程。每份 `.source.json` 包含程序、动作、物理轨迹和输出哈希及重放误差。

完整轨迹、模型、接触记录和报告在本机 `.esr-local/<场景>/`，不随 Git 提交。元数据中的本机来源路径用于追溯原记录，在其他机器上需要对应的实验归档。
