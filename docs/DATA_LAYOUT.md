# 当前数据入口与合并规则

本次按“更新记录覆盖旧记录，补跑合入原批次，核验后删除补充目录”的要求整理。没有启动新的仿真或模型调用。

## 自然更新数据：已完成合并

当前唯一审计入口是 `private/analysis/natural-safety-20261119/`，实际位于 `/mnt/data1/zhifeng/ESR-data/private/analysis/`。原始自然更新轨迹仍位于 `/mnt/data1/zhifeng/ESR-natural/20261118-r3-v50/`，没有修改。

| 当前记录 | 回合数 |
|---|---:|
| 已评估规则 PASS | 2721 |
| 已检出 VIOLATION | 997 |
| 明确未执行、不可评价 NOT_EVALUATED | 32 |

原 32 个整体 UNKNOWN 已按实际执行证据更新为 `NOT_EVALUATED`，其中 Nut 27、Wipe 5。具体原因直接写入每个 `episodes/<场景>/<链>/<版本_seed>.json` 的 `execution_reason` 和 `execution_diagnostic`，也写入统一 `episodes.csv`。未执行不能计为安全通过；载荷、力矩、卡滞等未校准风险仍逐规则 UNKNOWN。

扩展 v2 的物理判定及 Nut 的原因诊断已经合入，独立目录 `natural-safety-extended-v2-20261119/` 和 `nut-evidence-definitions-20261120/` 已删除。当前报告为 [REPORT.md](../private/analysis/natural-safety-20261119/REPORT.md)。15 条任务谱系、3750 个原始开发回合及父子退化计数不变。

按后续“删除不需要的审计历史文件”要求，已删除 7,595 个旧版逐回合审计缓存、Nut 子集重复文件、旧版本对照表和退役报告脚本，审计文件占用从约 642 MiB 降至 298 MiB。完成回执为主目录 `history-pruning.json`。当前原因直接读取主记录；附加规则定义保存在 `definitions/`，所用源码统一保存在 `frozen_sources/`。单位校验与缺测测试夹具保存在 `calibration/checks/` 和 `calibration/fixtures/`。

## 受控测量数据：只保留合并版

当前唯一数据入口为 `.esr-local/measurement-validation/controlled-20261120-r5/`，实际存储在 `/mnt/data1/zhifeng/ESR-data/`。它已合入有效补跑及 ESR v6 判定，共 **1524 个完整主记录**，另保留 **24 次失稳复跑证据**；复跑不增加有效主样本分母。最新公开图表统一位于 `assets/measurement_validation/controlled-20261120-r5/`。

按“只保留合并后的版本”要求，已清理 R1–R4、停止的 R6/R7、R8-recovery、四个结构复跑等待目录、旧日志、调试目录、旧图表、冗余源码压缩包、模型缓存和过期判定正文。R8、R9 及实际完成的结构复跑目录此前已吸收并删除。旧结果正文不再作为独立数据版本保存；部分必要来源仅留下路径及 SHA-256。

保留的内容为：

```text
controlled-20261120-r5/
├── protocol.json, source_hashes.json
├── source/                     原始物理程序及资源
├── analysis-capsules/           实际补跑和 v6 重审所需的冻结源码
├── calibration/                 校准轨迹与冻结限值
├── validation/                  1104 个诊断和额外案例
├── gate/                        360 个父子主循环回合及门控决定
├── replay/                      24 个原跨进程重放回合
├── fixtures/                    缺测、身份与边界夹具
├── requests/                    原始执行请求及日志
├── supplemental/               24 次失稳复跑的摘要与执行证据
├── report/                      最新汇总、表格和图片
├── verification/                验证记录
├── provenance/                  来源索引、执行源码映射与清理清单
├── history-pruning.json         精简结果及数据不变核验
└── consolidated-artifacts.json  当前文件完整哈希
```

Wipe 已完成定向修复：重新执行并合入 95 个 Wipe 回合（校准 3、验收 60、主循环 30、重放 2），使用 v7；主风险仍检查非工具机器人接触，新增的关节接近限位单独记录。主故障诊断 12/12、主循环候选 15/15 检出实际碰撞，全部 95 个回合参考 PASS。原失败注入的 28 条轨迹及其关节重审保存在 `supplemental/wipe-repair-v7/prior-primary/`，只作为解释原失败的补充证据。主记录总数仍为 1524，其他十一场景数据不变。修复回执为 `wipe-repair.json` 与 `verification/wipe-repair-v7.json`。过载与偏心力矩额外案例仍未全部触发，完整工程验收仍为 FAIL。

35 个回合实际使用过 R7 程序，相关源码已纳入 `analysis-capsules/`，R7 独立实验数据已删除。`provenance/execution-sources.json` 当前将 1314 个原程序回合、35 个 R7 补跑回合、80 个 R8 补跑回合和 95 个 Wipe 修复回合绑定到仍保留的源码与哈希清单。这些源码是合并版的复算依赖。

完整回合、当前测量结果、工程限值及最新报告在清理前后逐项核对。内容相同的不可变模型和轨迹块可以共享硬链接，文件路径、精度和采样字节保留；磁盘去重回执为 `storage-deduplication.json`。场景工程验收结果仍由原报告决定，清理不改变 FAIL、UNKNOWN 或检出率。

合并状态见 [任务状态](../.esr-local/_data-consolidation/20261120/status.json)；精简结果见主目录 [history-pruning.json](../.esr-local/measurement-validation/controlled-20261120-r5/history-pruning.json)。

## 来源与核验

合并后的主记录直接使用最新结果，不再要求叠加多个补充目录读取。每个主目录内：

- `provenance/source-index.json`：仍保留的来源文件及其当前存储位置。
- `provenance/pruned-index.jsonl.gz`：已删除旧判定的原路径和 SHA-256，仅保留元数据。
- 受控数据的 `provenance/objects/`：保留复算必需的原执行回执和失稳复跑证据；过期派生判定已精简。
- `provenance/absorption.json`：原合并时的独立哈希核验结果及合并程序哈希；同时注明后续精简政策。
- `consolidated-artifacts.json`：当前完整文件哈希。
- `cleanup.json`：已被吸收并清理的源目录。
- `history-pruning.json`：历史精简数量、占用和当前记录不变的核验结果。

原始物理执行请求及采样保留在原始数据入口。查询仍保留的验证来源时使用 `provenance/source-index.json`；已精简的旧审计判定正文不再提供，仅在 `pruned-index.jsonl.gz` 留存来源哈希。后续新实验应创建新批次。

窄范围合并工具为 `python -m esrb tools data consolidate`，只处理清单中的本批完成补充；计划见 [plan.json](../.esr-local/_data-consolidation/20261120/plan.json)。本批归并已结束，日常查看直接使用上述唯一数据入口。
