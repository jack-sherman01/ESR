# 自然技能更新的论文分析

`natural-study` 将原生任务决定与当前冻结的 ESR 逐回合审计配对，重建实际父子关系，生成任务习得、任务与违规交叉、两种退化口径、接纳谱系、连续物理量和源码证据案例。它不重新调用检测器，不修改冻结审计，也不执行模型或仿真。

```bash
.venv/bin/python -m esrb tools analyze natural-study \
  --campaign .esr-local/natural-campaigns/natural-seeds1to5-r3-v50-20261118 \
  --audit private/analysis/natural-safety-20261119 \
  --out private/papers/rss2027-stage1/natural-analysis \
  --plot-python /usr/bin/python3
```

输出目录必须为空，且不能与任何输入批次、链或审计目录重叠。绘图解释器需有 matplotlib；主分析使用项目 `.venv`。本机 `/usr/bin/python3` 可用于独立绘图，不需要改变原仿真环境。

交付入口为 `REPORT.md`，附带 `DATA_DICTIONARY.md`、层级 CSV 表、JSON 图表数据、六组 PNG/PDF 图、全部新增违规更新的程序 diff 与物理证据。`input-manifest.json` 记录读取文件的 SHA256，`validation.json` 记录身份、接纳条件与原始接触核对，`output-manifest.json` 绑定交付物。完整原始轨迹和载荷明细通过输入清单引用，避免重复复制。

源码复核说明绑定特定父子程序哈希；其他程序不会套用这些说明。该说明、Agent 理由和程序差异只能支持观察关联，不能替代反事实仿真。

任务通过与安全判定分别保存。`PASS` 限于冻结且已评估规则，`NOT_EVALUATED` 不计安全通过。逐规则新增违规与“父五 seed 均 PASS”的严格面板退化分别统计。首次建立任务父版不进入已有父版更新分母，失败候选不成为接纳谱系上的虚构父版。

输入为共享开发 seed 1–5 的十五条链，结论不外推到隐藏 seed，也不评价尚未执行的 CE 比较或在线 ESR 闭环效果。

针对统计与谱系的检查：

```bash
.venv/bin/python -m pytest -q tests/core/test_natural_study.py tests/workflows/test_natural.py
```
