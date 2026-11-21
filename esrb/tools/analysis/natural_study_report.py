"""Chinese paper-oriented report; conclusions are drawn from rebuilt tables."""
from collections import Counter

from .natural_study import LABELS


def md_table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |'] +
                     ['| '+' | '.join(str(x) if x is not None else '—' for x in row)+' |' for row in rows])


def conclusions(t, cases):
    total = next(s for s in t['scenes'] if s['task'] == 'ALL')
    scenes = [s for s in t['scenes'] if s['task'] != 'ALL']
    learned = [c for c in t['chains'] if c['first_qualified_version'] is not None]
    masked = [c for c in learned if c['final_clean_masks_earlier_accepted_violation']]
    success_violations = sum(r['task_qualified'] and r['audit_status'] == 'VIOLATION' for r in t['episodes'])
    new_cases = [c for c in cases if c['selection'] == 'new_violation']
    drops = [c for c in new_cases if {w['event']['kind'] for w in c['witnesses']} == {'drop'}]
    accepted_new = [c for c in new_cases if c['task_accepted']]
    accepted_contacts = sum(all('contact_row' in w['event'].get('contact', w['event']) for w in c['witnesses'])
                            for c in accepted_new)
    fast = sum(c['relative_endpoint_steps_change'] < 0 for c in learned)
    return [
        '## 7. 综合结论与论文表述', '',
        '**在当前五个开发 seed 和冻结规则范围内，任务驱动的 Agent 更新确实会产生新增违规，'
        '其中一部分仍满足任务要求，并成为后续版本的父策略。**', '',
        '### 7.1 数据直接支持的结论', '',
        '**（1）任务成功不足以判断程序是否符合物理安全规则。** '
        f'{total["task_qualified_episodes"]} 个任务合格回合中，{success_violations} 个同时存在冻结规则违规；'
        f'{total["accepted_candidates"]} 个任务接纳程序中，{total["accepted_unsafe_candidates"]} 个至少在一个开发 seed 上违规。'
        '任务结果与物理审计提供不同信息，单看成功率会遗漏成功过程中的碰撞等事件。'
        '含违规程序数包括初始已有及持续保留的违规，不能全部计为新增退化或独立事故。', '',
        '**（2）已经通过当前审计面板的父策略，修改后仍可能退化。** '
        f'父版五个 seed 全部为冻结规则 PASS 的更新中，有 {total["strict_new_violation_updates"]} 个子版出现违规，'
        f'其中 {total["accepted_strict_new_violation_updates"]} 个仍通过五 seed 任务资格并替换父版。'
        '这些严格配对案例直接说明，父版通过检查并不能保证后续任务合格修改保持通过，支持对更新逐次检查的必要性。'
        '这里的 PASS 仅覆盖已评估规则，不代表所有物理风险均被排除。', '',
        '**（3）任务选择对不同危险的约束作用不同。** '
        f'本批新增违规更新中，{len(drops)} 个掉落候选有 {sum(not c["task_accepted"] for c in drops)} 个因任务不合格而被拒绝；'
        f'{len(accepted_new)} 个被接纳的新增违规更新中，{accepted_contacts} 个具有接触或碰撞证据。'
        '这提示，破坏任务完成的危险可能被任务标准间接排除，与任务成功共存的危险则可能被保留。'
        '拒绝来自当时的任务资格选择，不能写成 ESR 在运行时阻止了危险；也不能推断所有掉落或碰撞都遵循这一规律。', '',
        '**（4）只检查最终版本，会遗漏更新过程中的问题。** '
        f'{len(masked)} 条链最终父面板 PASS，却曾接纳过含违规版本：'
        + '、'.join(f'{LABELS[c["task"]]} 第 {c["chain"]} 链' for c in masked) + '。'
        '实际谱系还观察到逐规则违规消失后再次出现。因此，最终版本通过不能说明整条链一直通过；'
        '需要保留首次出现、持续、修复与复发的历史。', '',
        '**（5）小幅程序修改也可能伴随物理违规变化。** '
        '源码复核中的修改包括降低几毫米搬运净空、减少中间路径点、增大运动分段间距，以及合并最后下压动作；'
        '相应子版执行中观察到新增掉落或碰撞。'
        f'另外，{fast}/{len(learned)} 条已习得链的最终父版控制步更少。'
        '这些是程序变化、效率和物理事件之间的观察关联，尚不能证明追求效率导致安全退化。', '',
        '**（6）不同场景呈现不同退化现象，目前不能据此排列其固有安全性。**', '',
        md_table(['场景', '新增逐规则违规更新', '其中被任务接纳'],
                 [[LABELS[s['task']], s['new_violation_updates'], s['accepted_new_violation_updates']] for s in scenes]), '',
        '例如，PickPlaceCan 的新增违规更新均通过任务选择，体现了任务成功与接触违规共存的情况。'
        'Wipe 的零计数则必须结合任务习得不足、实际活动和规则覆盖解释，不能写成 Wipe 最安全。'
        '每场景仅三条生成链，不能将相邻版本和大量物理步当作独立重复来推断场景总体差异。', '',
        '### 7.2 机制解释与待验证假设', '',
        '一种可能解释是：任务反馈对导致失败的掉落较敏感，对不妨碍完成任务的碰撞缺少直接选择压力；'
        '缩短路径、减少动作或调整抓持时，物理交互余量也可能改变。'
        '当前案例、Agent 理由与源码差异提供研究线索，但没有通过修改消融、反事实执行或随机化效率压力分离其因果作用。', '',
        '### 7.3 可用于论文的核心表述与结论边界', '',
        '> 在五场景自然技能更新实验中，任务合格不能保证冻结物理规则的保持；新增违规可以通过任务选择进入接纳谱系，'
        '而终版检查可能遗漏中间版本的违规历史。', '',
        '这些结果为 ESR 所针对的问题提供了真实案例与可追溯证据。'
        '**本批没有证明在线接入 ESR 能减少后续退化，也没有证明 CE 比固定测试或均匀测试更有效。** '
        '新 seed 泛化、开发 seed 数量的作用、真实损伤后果与未校准风险，以及在线门控改变后续提案的效果，'
        '均需要另行实验。以上结论只适用于当前开发初态、生成协议和冻结且已评估的物理规则。', '',
    ]


def report(t, cases, checks, campaign, audit):
    total = next(s for s in t['scenes'] if s['task'] == 'ALL')
    scene_rows = [s for s in t['scenes'] if s['task'] != 'ALL']
    rates = {(r['task'], r['metric']): r for r in t['rates']}
    def ratio(name):
        r = rates[('ALL', name)]
        return f'{r["numerator"]}/{r["denominator"]}（{100*r["value"]:.1f}%）' if r['denominator'] else '无可判定分母'
    updates = t['updates']; accepted = [u for u in updates if u['task_accepted']]
    learned = [c for c in t['chains'] if c['first_qualified_version'] is not None]
    fast = [c for c in learned if c['relative_endpoint_steps_change'] < 0]
    unsafe_success = sum(r['task_qualified'] and r['audit_status'] == 'VIOLATION' for r in t['episodes'])
    nonexecuted = Counter((r['task'], r['execution_reason']) for r in t['episodes'] if r['audit_status'] == 'NOT_EVALUATED')
    intervals = t['lineage_intervals']
    strict_cases = [c for c in cases if c['selection'] == 'new_violation' and c['task_accepted'] and c['all_five_parent_pass']]
    new_cases = [c for c in cases if c['selection'] == 'new_violation']
    final_bad = sum(c['final_parent_violation_cases'] > 0 for c in learned)
    initial_bad = sum(c['first_parent_violation_cases'] > 0 for c in learned)
    masked = [c for c in learned if c['final_clean_masks_earlier_accepted_violation']]
    generation = t['generation_usage']
    unrecorded_usage = sum(not g['usage_recorded'] for g in generation)
    native_disagreement = [r for r in t['episodes'] if r['native_task_success'] and not r['task_qualified']]
    lines = ['# 五场景 Agent 自然技能更新与 ESR 离线审计分析', '',
        '本报告分析已经发生的任务驱动更新，并把其任务结果与 ESR 事后物理审计配对。'
        'ESR 没有参与这些 Agent 的反馈、接纳或停止决策。所有“通过”均限定于当前冻结且可评估的规则。', '',
        '## 1. 核心结果与证据边界', '',
        f'- {total["chains"]} 条链、{total["candidates"]} 个候选、{total["episodes"]} 个开发回合；'
        f'{len(learned)} 条链建立任务合格父版，{len(t["chains"])-len(learned)} 条未建立。',
        f'- 任务合格回合 {ratio("task_qualification")}；其中 {unsafe_success}/{total["task_qualified_episodes"]}'
        f'（{100*unsafe_success/total["task_qualified_episodes"]:.1f}%）同时存在冻结规则违规。任务成功不能替代物理审计。',
        f'- {total["accepted_unsafe_candidates"]}/{total["accepted_candidates"]} 个任务接纳程序至少在一个开发 seed 出现违规。'
        '这包括初始父版已有违规及后续持续违规，不能全部称为新增安全退化。',
        f'- 在 {len(updates)} 个具有实际任务父版的更新中，{sum(u["new_violation"] for u in updates)} 个观察到新增逐规则违规；'
        f'其中 {sum(u["new_violation"] for u in accepted)} 个被任务选择接纳。',
        f'- 严格五 seed 父面板 PASS 的可判定更新中，新增违规为 {ratio("strict_panel_new_violation")}；'
        f'其中 {len(strict_cases)} 个被接纳。此口径与逐规则退化分开报告。',
        f'- {len(fast)}/{len(learned)} 条已习得链的最终父版比首个合格父版平均控制步更少。'
        '本实验没有随机化效率压力，不能据此断言效率改善造成违规。', '',
        '不能由本数据推出：新 seed 上的总体安全概率、开发 seed 数量的因果作用、CE 相对其他选择器的优势，'
        '或在线 ESR 对后续 Agent 演化的改善。', '',
        '## 2. 数据、协议与质量', '',
        f'自然批次：`{campaign}`。', '', f'冻结审计：`{audit}`。', '',
        '每场景三条独立生成链，每链 50 个候选；每候选固定使用开发 seed 1–5。'
        '三条链共享相应开发初态，独立性来自生成过程；五个 seed 及相邻版本均不作为独立更新链重复。'
        '“Agent 学习”指技能程序迭代，不是模型权重训练。', '',
        md_table(['场景', '任务合格回合/750', '接纳程序/150', '有父版链/3', 'PASS', 'VIOLATION', 'NOT_EVALUATED'],
                 [[LABELS[s['task']], s['task_qualified_episodes'], s['accepted_candidates'], s['chains_with_task_parent'],
                   s['pass_episodes'], s['violation_episodes'], s['not_evaluated_episodes']] for s in scene_rows]), '',
        f'整体标签：{total["pass_episodes"]} PASS、{total["violation_episodes"]} VIOLATION、'
        f'{total["unknown_episodes"]} UNKNOWN、{total["not_evaluated_episodes"]} NOT_EVALUATED。'
        '无整体 UNKNOWN 不表示所有风险项均可判定；未校准载荷、力矩、卡滞等仍保持逐项未知。', '',
        f'严格任务资格与原生成功有 {len(native_disagreement)} 个分歧回合：原生成功但程序未满足完成资格。'
        '回合表同时保存过程任意采样成功、终态成功、程序状态和最终资格。', '',
        md_table(['场景', '未执行原因', '回合数'], [[LABELS[k[0]], k[1], n] for k, n in sorted(nonexecuted.items())]), '',
        '这些回合保留在全部回合与任务失败分母中，排除于可判定安全分母。'
        '并行负载使墙钟时间受资源竞争影响，控制步数用于策略效率比较；墙钟与 token 使用只作实际开销描述。', '',
        f'记录物理步合计 {sum(r["physics_steps"] for r in t["episodes"]):,}；'
        f'本次重新核对 {sum(c["status"] == "PASS" for c in checks)} 个案例接触证据行的部件、位置、原始 wrench 及坐标变换。'
        '完整轨迹采样检查沿用冻结审计；本步骤没有重新扫描所有物理步或重新认证检测器。', '',
        f'从当前逐回合标签重建汇总，与旧派生缓存比较发现 {len(t["cached_table_differences"])} 处字段差异，'
        '详见 tables/cached_table_differences.csv；原缓存与冻结审计均未覆盖。', '',
        '### 中途上下文协议变化', '',
        '因累计模型输入过长，生成过程从完整历史切换为当前父程序和近期任务反馈。'
        '各链切换位置如下，图中黑色刻线对应此变化；无法用本批前后变化估计其因果效果。', '',
        md_table(['场景', '链', '当前协议记载起始版本'], [[LABELS[c['task']], c['chain'], c['context_revision_effective_from']] for c in t['chains']]), '',
        'Wipe 第三链 V14 的上下文修改涉及槽位内重试；完整尝试配置见 generation_usage.csv，不能把每槽位假设成完全相同生成协议。', '',
        f'保存的生成尝试 {len(generation)} 次，缺完成用量记录 {unrecorded_usage} 次；'
        f'可见 input tokens {sum(g["input_tokens"] or 0 for g in generation):,}、'
        f'output tokens {sum(g["output_tokens"] or 0 for g in generation):,}。'
        '失败尝试保留，缺用量不补零解释为免费；requested_model 不是供应商底层快照的独立证明。', '',
        '## 3. Agent 的任务习得与效率', '',
        md_table(['场景', '链', '首个合格版本', '接纳数', '最终父版', '首/末平均控制步', '首/末违规 seed 数'],
                 [[LABELS[c['task']], c['chain'], c['first_qualified_version'], c['task_accepted'], c['final_parent_version'],
                   f'{c["first_parent_mean_steps"]:.1f}/{c["final_parent_mean_steps"]:.1f}' if c['first_parent_mean_steps'] else '—',
                   f'{c["first_parent_violation_cases"]}/{c["final_parent_violation_cases"]}' if c['first_qualified_version'] else '—'] for c in t['chains']]), '',
        '未建立合格父版的链保留全部失败候选和物理事件，但不进入已有合格父版后的退化分母。'
        '初始合格版本属于任务习得阶段，不能计为相对已有父版的退化。', '',
        '逐 seed 结果见 tables/seed_performance.csv，失败终止来源见 tables/failure_reasons.csv。'
        '这些数据用来区分单一困难初态与五 seed 普遍失败，不构成未见初态泛化评估。', '',
        '![全版本演化](figures/version_evolution.png)', '',
        '上两层分别是候选的任务资格与违规 seed 数；方框表示含未执行或未知回合，零违规不能覆盖它们。'
        '第三层为候选处理后的实际父版，空白表示尚未习得合格父版。', '',
        '![父版效率](figures/task_parent_efficiency.png)', '',
        '端点比较使用各链首个与最终合格父版的同五个成功初态，非简单比较 V1 与 V50。', '',
        '## 4. 任务成功与违规的交叉关系', '',
        '![任务与审计结果](figures/task_audit_cross.png)', '',
        md_table(['指标', '分子/分母及比例'], [[name, ratio(key)] for name, key in [
            ('可判定回合中观察到违规', 'episode_violation_assessable'),
            ('候选至少一个 seed 观察到违规', 'candidate_with_observed_violation'),
            ('任务接纳程序至少一个 seed 观察到违规', 'accepted_candidate_with_observed_violation'),
            ('所有有父版提案中观察到逐规则新增违规', 'new_predicate_violation_observed_all_updates'),
            ('任务接纳后续更新中观察到逐规则新增违规', 'accepted_update_with_new_predicate_violation')]]), '',
        '候选级“观察到违规”是有限开发面板上的事实，不是总体风险估计；没有事件也不能覆盖未评估风险。'
        '每种 hazard 分别给出涉及回合、候选和链的去重计数，见 tables/hazard_types.csv。'
        '不同 hazard 可能来自同一接触，不能把表中行相加作为事故总数。', '',
        '### Wipe 零离散违规的解释', '',
        'Wipe 的零计数仅针对已启用规则。该场景只有一条链建立五 seed 合格父版，另两条链的任务活动与覆盖必须一起查看。'
        '预期工具—桌面接触属于任务动作；原生 60 N 末端传感器范数约束不等于硬件损伤限值。'
        '未校准外载、力矩、卡滞和未来稳定性不能由零事件推为安全。连续工具载荷见 continuous_summary.csv 与 body_interface_summary.csv。'
        '完整原始载荷表仍由输入清单指向冻结审计，不重复复制。', '',
        '## 5. 新增退化、遗传与修复', '',
        md_table(['场景', '有父版更新', '逐规则新增', '其中接纳', '严格父面板 PASS', '严格新增', '其中接纳'],
                 [[LABELS[s['task']], s['parent_updates'], s['new_violation_updates'], s['accepted_new_violation_updates'],
                   s['strict_clean_parent_updates'], s['strict_new_violation_updates'], s['accepted_strict_new_violation_updates']] for s in scene_rows]), '',
        '逐规则退化要求父在同一初态、同一具体规则可判定 PASS 而子为 VIOLATION；父可能在其他 seed 或其他规则已有违规。'
        '严格口径额外要求父面板五回合均为冻结规则 PASS。两者都不把任务合格自动当成安全合格。', '',
        f'{initial_bad}/{len(learned)} 条已习得链的首个任务父版已有违规；最终父版中 {final_bad}/{len(learned)} 条仍有违规。'
        f'{len(masked)} 条链最终父面板 PASS 却曾接纳过含违规版本：'
        + '、'.join(f'{LABELS[c["task"]]} C{c["chain"]}' for c in masked) + '。', '',
        f'实际接纳谱系得到 {len(intervals)} 段 seed×predicate 违规区间，其中 '
        f'{sum(r["resolution_version"] is not None for r in intervals)} 段之后观察到该规则 PASS，'
        f'{sum(r["recurrence_after_known_pass"] for r in intervals)} 段为已知 PASS 间隔后的复发。'
        '粗粒度规则与部件规则有重叠，这些区间不是独立事故数；持续代数按接纳父状态计，末尾未修复为右删失。', '',
        '![接纳谱系](figures/accepted_lineages.png)', '',
        '连线只连接实际接纳父版。灰色叉号是被拒提案，蓝圈表示新增逐规则违规。'
        '同样的违规 seed 数可能掩盖 hazard 类型替换，应结合 paired_cases.csv 的逐规则变化。', '',
        '![配对效率与违规变化](figures/paired_efficiency.png)', '',
        '每点为一个父子更新；步数仅在父子共同成功 seed 上比较。纵轴是违规 seed 数变化，不是严重度；'
        '横轴左移也不表示导致右侧物理事件的原因。', '',
        '## 6. 程序变化与物理证据案例', '',
        f'全部 {len(new_cases)} 个新增违规更新均生成父子源码 diff、Agent 原始修改理由、五 seed 配对结果、'
        'API 时间线与物理证据索引；另外按“场景内链号/版本最早”的固定规则选择持续与修复对照。'
        '规则标签是事后产生，不能写成 Agent 当时已经得到安全反馈。', '',
        md_table(['场景/链', '父→子', '任务接纳', '父面板全 PASS', '新增规则', '案例'],
                 [[f'{LABELS[c["task"]]} C{c["chain"]}', f'V{c["parent_version"]}→V{c["version"]}',
                   c['task_accepted'], c['all_five_parent_pass'], ', '.join(c['new_predicates']),
                   f'[证据](cases/{c["id"]}.md)'] for c in new_cases]), '',
        '![严格退化案例证据](figures/regression_evidence.png)', '',
        '图示实际 API 动作区间和首个已核验接触证据；不是完整力曲线，也不把首样本力解释为峰值或严重程度。'
        '完整程序差异见各案例 .diff，部件身份、原始 chunk 行号及五 seed 身份见 cases.json。', '',
        '### 源码复核摘要', '',
        md_table(['更新', '已核对的程序变化与观察结果'],
                 [[f'{LABELS[c["task"]]} C{c["chain"]} V{c["parent_version"]}→V{c["version"]}', c['source_review']] for c in new_cases]), '',
        '### 观察与解释分开', '',
        '- 数据支持：任务选择可以接纳仍然完成任务的碰撞候选，也可以因任务失败拒绝掉落候选。'
        '具体数量以案例和表格为准，不能因此把所有掉落或所有碰撞泛化成必然行为。',
        '- 可能解释：部分程序为减少补抓、缩短路径或调整抓持而改变空间余量；'
        'Agent 理由和源码差异提供线索，不能替代受控反事实执行。',
        '- 尚待实验：同一修改在更多初态的稳定作用、违规的真实损伤后果，以及 ESR 在线接入后 Agent 如何重新提案。', '',
        *conclusions(t, cases),
        '## 8. 可复算交付与后续实验', '',
        'input-manifest.json 绑定全部读取的原生任务记录、程序、初态、模型、审计和证据；validation.json 保存输入不变性及证据检查。'
        'output-manifest.json 绑定交付文件。analysis.json 是图表输入，tables/ 提供逐回合到场景的层级表，DATA_DICTIONARY.md 定义口径。', '',
        '复算命令（输出必须是独立空目录）：', '', '```bash',
        '.venv/bin/python -m esrb tools analyze natural-study \\',
        f'  --campaign {campaign} \\', f'  --audit {audit} \\',
        '  --out <独立空目录> --plot-python /usr/bin/python3', '```', '',
        '后续若要评价 CE，需另行冻结门控面板、独立参考面板与预算，按时间顺序只向选择器暴露已查询结果。'
        '当前分析可用于选择案例与提出假设，不能充当未经执行的参考标签。'
        '若要评价在线 ESR 对演化的影响，需要真正改变父版接纳后重新生成后代，不能离线删除历史违规版本模拟。', '']
    return '\n'.join(lines)


def dictionary():
    return '''# 数据字典与统计口径

所有表以 task、chain、version、seed 中适用的字段联合定位。布尔量在 CSV 中为 True/False；空值表示未知或不适用，不能补零。

| 文件 | 单位与口径 |
|---|---|
| episodes.csv | 一次开发执行；原生过程成功、终态成功、程序完成与严格任务资格分开。NOT_EVALUATED 仍计任务结果，不计可判定安全分母。 |
| candidates.csv | 一个候选槽位；generation_base 是生成起点，parent_version 是实际任务父版，二者不可互换。 |
| updates.csv | 生成时已有任务父版的候选；new_violation 是同规则 PASS→VIOLATION，strict_panel_new_violation 另要求父五 seed 均为冻结规则 PASS。 |
| paired_cases.csv | 实际父子在相同实际初态及模型上的一个 seed 配对；效率仅在 common_task_success 时比较。 |
| chains.csv | 一条生成链；首尾使用首个/最终任务合格父版。尚未建立父版为空值。 |
| seed_performance.csv / failure_reasons.csv | 每链每 seed 的任务结果与原始终止原因；用于定位困难初态。 |
| scenes.csv / rates.csv | 场景汇总及明确分子、分母；ALL 仅描述本数据集，不代表任务总体分布。 |
| task_audit_cross.csv | 任务资格×审计状态计数，所有状态均保留。 |
| hazard_types.csv | 各类事件涉及的唯一回合/候选/链数；规则之间可重叠，不能加和为事故次数。 |
| lineage_intervals.csv | 实际接纳谱系的 seed×predicate 连续违规区间；规则重叠，不是独立事故。未知打断连续区间，末尾为右删失。 |
| generation_usage.csv | 所有保存的生成尝试，包括静态失败/重试；缺 token 记录保持空。requested_model 不等于已验证底层模型快照。 |
| continuous_changes.csv | 冻结连续测量的父子变化，力 N、时间 s；没有动态证据的配对为 NOT_COMPARABLE，不用于连续变化汇总。 |
| continuous_summary.csv | 相同场景/接触界面/通道/测量/单位内的连续变化；不合并为无量纲风险分数。 |
| body_interface_summary.csv | 将同一回合/部件界面各阶段的峰值取最大、活动时间相加后，统计有界面记录的回合分布；力 N、body origin 力矩 Nm。完整原表由 input-manifest.json 引用，不复制。 |
| paired_body_moments.csv | 同初态父子、同 body origin 界面的峰值力矩变化 Nm；仅完整动态轨迹，未出现的接触界面记零，不产生过载判定。 |
| rule_coverage.csv | 按场景逐规则的 PASS/VIOLATION/UNKNOWN/NOT_EVALUATED 计数，以及未评估风险；PASS 只覆盖该规则。 |
| data_quality.csv | 输入身份、覆盖、未执行和任务判定分歧的数量。 |
| cached_table_differences.csv | 当前逐回合重建值与旧派生表的差异；原审计未修改。 |
| cases.json / cases/ | 21 个新增违规更新及固定选择的持续/修复对照；源码 diff、理由、配对结果、API 时间线和原始接触证据。 |

PASS = 冻结且已评估规则未观察到违规；VIOLATION = 至少一条规则有证据；UNKNOWN = 证据不足；NOT_EVALUATED = 未执行物理动作。无整体 UNKNOWN 不代表未评估风险已通过。

新增逐规则违规的总体分母包含全部已有父版更新，含义是“观察到新增违规的更新比例”，不是对未覆盖风险的可判定退化率。严格面板比例的分母为父五 seed PASS 且子面板有违规或全部 PASS 的更新。

回合、候选和更新共享初态及谱系，不能当作独立链重复。只报告链级及场景级描述统计，不从三条链构造夸大精度的显著性结论。
'''
