"""Export and check existing records only; never execute a policy or simulation."""

# Allow direct execution from this subdirectory as well as package imports.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import csv
import difflib
import hashlib
import json
from pathlib import Path
import re
import shutil
import statistics
import sys
from datetime import datetime, timezone

import argparse
import importlib
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root', type=Path, required=True)
ROOT = parser.parse_args().root.resolve()
REPO = Path(__file__).resolve().parents[2]
task = json.loads((ROOT / 'protocol.json').read_text())['task']
module_names = {'two_arm_handover': ('handover', 'TwoArmHandover'), 'spill_wipe': ('wipe', 'Wipe'), 'nut_assembly': ('nut', 'NutAssemblySquare')}
prefix, environment_name = module_names[task]
state = importlib.import_module('scripts.gates.' + prefix + '_gate_state')
chain_runner = importlib.import_module('scripts.runners.run_' + prefix + '_long_chain')

def read(path):
    return json.loads(path.read_text())

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')

summary = chain_runner.summarize(ROOT)
assert summary['status'] == 'COMPLETE' and summary['measured'] == 50
assert summary['unique_source_programs'] == summary['unique_ast_programs'] == 50
assert summary['actual_initial_identities'] == 1 and summary['all_sampling_complete']
assert summary['unknown'] == 0
state.load_rules(ROOT)
chain = state.load_chain(ROOT, 1)
assert len(chain['events']) == 50
assert read(ROOT / 'round_01/fresh_agent_complete.json')['measured'] == 50
contract = read(ROOT / 'worker_contract.json')
capsule = Path(contract['capsule'])
shutil.copytree(capsule, ROOT / 'worker_capsule', dirs_exist_ok=True)
reference = read(ROOT / 'round_01/initial_reference.json')
protocol = read(ROOT / 'protocol.json')
protocol_sha = sha(ROOT / 'protocol.json')
rows = list(csv.DictReader((ROOT / 'episodes.csv').open()))
results, decisions, unsafe, regressions = [], [], [], []
checks = []
provided = [(ROOT / 'worker_capsule/API.txt', (capsule / 'API.txt').read_text())]
for extra in ['worker_launch_instructions.txt', 'worker_host_updates.json']:
    if (ROOT / extra).exists():
        provided.append((ROOT / extra, (ROOT / extra).read_text()))
leaks = []
pattern = re.compile(r'esrb|\bsafety\b|\bgate\b|violation|forbidden_contact|parent_before|parent_after|normal_force|raw_sensor|门控|安全规则', re.I)
critical = [ROOT / p for p in ['protocol.json', 'safety_rules.json', 'worker_contract.json',
    'round_01/initial_reference.json', 'round_01/gate_chain.json', 'round_01/active_parent.json',
    'round_01/agent_assignment.json', 'round_01/fresh_agent_complete.json']]
for version in range(1, 51):
    folder = ROOT / f'round_01/version_{version:02d}'
    generation = ROOT / f'round_01/generation_{version:02d}'
    result = read(folder / 'execution/result.json')
    decision = read(folder / 'gate_decision.json')
    proposal = read(folder / 'proposal.json')
    assert read(generation / 'task_feedback.json') == read(capsule / f'feedback_v{version:02d}.json')
    assert read(generation / 'task_feedback.json')['api_diagnostics'] == read(capsule / f'additional_feedback_v{version:02d}.json')
    gen = read(folder / 'generation.json')
    agent_input = read(generation / 'agent_input.json')
    stem = f'v{version:02d}_attempt{gen["attempt"]:02d}'
    submitted = ROOT / 'round_01/submissions' / stem
    program_sha = sha(folder / 'program.py')
    assert program_sha == decision['candidate_sha256'] == proposal['source_sha256'] == result['program_sha256']
    assert program_sha == sha(generation / 'agent_output.py') == sha(submitted / 'program.py') == sha(capsule / f'{stem}.py')
    assert sha(folder / 'execution/result.json') == decision['result_sha256']
    assert sha(folder / 'execution/safety_physics.jsonl.gz') == result['safety']['witness_sha256']
    assert sha(folder / 'execution/model.xml') == reference['model_sha256']
    assert result['seed'] == reference['seed'] == summary['seed']
    assert result['initial_state_id'] == reference['initial_state_id']
    assert result['protocol_sha256'] == protocol_sha
    assert decision == chain['events'][version - 1]
    assert gen['agent_input_sha256'] == sha(generation / 'agent_input.json')
    assert gen['api_sha256'] == contract['api_sha256'] == hashlib.sha256(agent_input['api'].encode()).hexdigest()
    assert gen['fork_turns'] == 'none' and not gen['initial_program_provided']
    assert sha(folder / 'candidate_prompt.txt') == proposal['candidate_prompt_sha256']
    assert sha(folder / 'task_feedback.json') == proposal['task_feedback_sha256']
    if version == 1:
        assert agent_input['starting_program'] is None and agent_input['previous_task_feedback'] is None
    else:
        assert agent_input['previous_task_feedback'] == read(capsule / f'feedback_v{version-1:02d}.json')
        assert agent_input['starting_program'] == (capsule / f'starting_v{version:02d}.py').read_text()
    provided.extend([(generation / 'agent_input.json', json.dumps({key: agent_input[key] for key in ['task_prompt', 'api', 'previous_task_feedback']}, ensure_ascii=False)),
                     (folder / 'task_feedback.json', (folder / 'task_feedback.json').read_text()),
                     (generation / 'task_feedback.json', (generation / 'task_feedback.json').read_text()),
                     (capsule / f'additional_feedback_v{version:02d}.json', (capsule / f'additional_feedback_v{version:02d}.json').read_text())])
    checks.append({'version': version, 'source_sha256': program_sha, 'result_sha256': decision['result_sha256'],
                   'physics_witness_sha256': result['safety']['witness_sha256'], 'status': 'PASS'})
    for event in result['safety']['events']:
        geometries = {g['id']: g for g in read(folder / 'execution/geometry_groups.json')}
        enriched = {'version': version, 'parent_before': (decision['parent_before'] or {}).get('version'),
                    'retained_parent': decision['retained_parent'], 'observed_safety_regression': decision['observed_safety_regression'],
                    'event': event}
        if 'contact' in event:
            enriched['geometries'] = [geometries[event['contact'][key]] for key in ['geom1', 'geom2']]
        unsafe.append(enriched)
        if decision['observed_safety_regression']:
            regressions.append(enriched)
            parent = ROOT / decision['parent_before']['program_path']
            diff_dir = ROOT / 'regression_diffs'
            diff_dir.mkdir(exist_ok=True)
            (diff_dir / f'v{enriched["parent_before"]:02d}_to_v{version:02d}.diff').write_text(''.join(difflib.unified_diff(
                parent.read_text().splitlines(keepends=True), (folder / 'program.py').read_text().splitlines(keepends=True),
                fromfile=f'V{enriched["parent_before"]}', tofile=f'V{version}')))
    results.append(result)
    decisions.append(decision)
    critical.extend([folder / name for name in ['program.py', 'candidate_prompt.txt', 'task_feedback.json', 'generation.json',
        'proposal.json', 'execution_settings.json', 'gate_decision.json', 'execution/result.json', 'execution/model.xml',
        'execution/initial_state.npy', 'execution/safety_physics.jsonl.gz', 'execution/geometry_groups.json', 'execution/artifacts.json']])
    critical.extend(p for p in generation.iterdir() if p.is_file())
    critical.extend(p for p in submitted.iterdir() if p.is_file())
for path, content in provided:
    matches = sorted(set(m.group(0) for m in pattern.finditer(content)))
    if matches:
        leaks.append({'path': str(path), 'matches': matches})
assert not leaks, leaks
assert sha(capsule / 'API.txt') == contract['api_sha256']
qualified = [r for r, d in zip(results, decisions) if d['decision'] == 'PROMOTE']
best = min(qualified, key=lambda r: r['control_steps'], default=None)
first = qualified[0] if qualified else None
last = next((r for r in results if r['version'] == summary['active_parent']), None)
blocks = []
for lower in range(1, 51, 10):
    block = decisions[lower-1:lower+9]
    blocks.append({'versions': f'V{lower}–V{lower+9}', 'promotions': sum(d['decision'] == 'PROMOTE' for d in block),
                   'safety_rejections': sum(d['safety_status'] == 'VIOLATION' for d in block),
                   'task_rejections': sum(d['reason'] == 'task_not_completed' for d in block)})
# Reason strings are retained verbatim; count task-only rejections by exclusion.
for block, lower in zip(blocks, range(1, 51, 10)):
    block['task_rejections'] = sum(d['decision'] != 'PROMOTE' and d['safety_status'] != 'VIOLATION' for d in decisions[lower-1:lower+9])
analysis = {**summary, 'task_rejections': sum(d['decision'] != 'PROMOTE' and d['safety_status'] != 'VIOLATION' for d in decisions),
    'program_ok': sum(r['program_ok'] for r in results),
    'native_success_safety_rejection_versions': [r['version'] for r, d in zip(results, decisions) if r['native_task_success'] and d['safety_status'] == 'VIOLATION'],
    'first_qualified_version': (first['version'] if first else None), 'first_qualified_control_steps': (first['control_steps'] if first else None),
    'qualified_parent_candidates': sum(d['parent_before'] is not None for d in decisions),
    'best_qualified_version': (best['version'] if best else None), 'best_qualified_control_steps': (best['control_steps'] if best else None),
    'final_parent_control_steps': (last['control_steps'] if last else None),
    'physics_intervals': sum(r['safety']['physics_steps'] for r in results),
    'episode_wall_seconds_median': statistics.median(r['wall_seconds'] for r in results),
    'episode_wall_seconds_sum': sum(r['wall_seconds'] for r in results),
    'blocks': blocks, 'regression_versions': sorted({e['version'] for e in regressions}),
    'physics_reexecution': False}
save(ROOT / 'analysis.json', analysis)
save(ROOT / 'regression_cases.json', regressions)
save(ROOT / 'unsafe_events.json', unsafe)
save(ROOT / 'generation_input_audit.json', {'status': 'PASS', 'fork_turns': 'none', 'initial_program': None,
    'initial_previous_feedback': None, 'host_provided_text_records_checked': len(provided), 'leaks': leaks,
    'method': 'Check host task/API/public task feedback/API diagnostics for gate terms; exclude worker-authored source and rationale.',
    'scope': 'Recorded inputs and declared filesystem-access restriction; not OS-enforced isolation.'})
save(ROOT / 'validation.json', {'status': 'PASS', 'rules_and_implementation_hashes': 'PASS', 'gate_chain': 'PASS',
    'versions': checks, 'tests': {'reported_passed': read(ROOT / 'experiment_context.json').get('tests_passed')},
    'scope': 'Existing record integrity only; no independent simulation retest or offline physics reassessment.'})


lines = [f'# {environment_name}：单轮50版结果', '',
    f'固定 seed **{summary["seed"]}**，完成50个源码及AST均不同的技能版本。一个新的空上下文 Agent 从零编写V1，随后只根据本轮反馈修改至V50，成功后仍测满；不做独立复测。', '',
    '| 结果 | 数量 |', '| --- | ---: |',
    f'| 原生任务成功 | {summary["native_successes"]}/50 |',
    f'| 通过门控并晋升 | {summary["promotions"]}/50 |',
    f'| 安全拒绝 | {summary["safety_rejections"]}/50 |',
    f'| 任务拒绝 | {analysis["task_rejections"]}/50 |',
    f'| 已有合格父策略后的安全退化 | {summary["observed_safety_regressions"]} |',
    f'| 程序解释器完成 | {analysis["program_ok"]}/50 |',
    f'| UNKNOWN | {summary["unknown"]} |', '']
if analysis['native_success_safety_rejection_versions']:
    lines += ['原生成功但被安全规则拒绝的版本：' + '、'.join('V'+str(v) for v in analysis['native_success_safety_rejection_versions']) + '。这些版本不计晋升。', '']
if qualified:
    lines += [f'首个合格父策略 **V{first["version"]}**（{first["control_steps"]}控制步）；之后共有{analysis["qualified_parent_candidates"]}个已有父策略的候选。最终父版 **V{summary["active_parent"]}**（{last["control_steps"]}步），控制步最少的合格版为 **V{best["version"]}**（{best["control_steps"]}步）。门控不要求步数单调减少。', '']
else:
    lines += ['本轮没有建立合格父策略，不能把父子安全退化计数为零解释为更新安全。', '']
lines += ['| 版本 | 晋升 | 安全拒绝 | 任务拒绝 |', '| --- | ---: | ---: | ---: |']
lines += [f'| {b["versions"]} | {b["promotions"]} | {b["safety_rejections"]} | {b["task_rejections"]} |' for b in blocks]
if task == 'spill_wipe':
    progress = [{'version': r['version'], 'cleaned_markers': (r.get('last_telemetry') or {}).get('wiped_markers'), 'total_markers': (r.get('last_telemetry') or {}).get('num_markers')} for r in results]
    save(ROOT / 'task_progress.json', progress)
    numeric = [r for r in progress if type(r['cleaned_markers']) is int]
    if numeric:
        peak = max(numeric, key=lambda r: r['cleaned_markers'])
        lines += ['', f'最高原生清理进度：V{peak["version"]}，{peak["cleaned_markers"]}/{peak["total_markers"]}个标记。']
if unsafe:
    lines += ['', '安全事件：力为停止时首次接触采样值，并非完整碰撞过程峰值，未用于过载分类。', '',
        '| 候选 | 已有父版 | 事件/部件 | 仿真时间/s |', '| --- | --- | --- | ---: |']
    for item in unsafe:
        event = item['event']
        pair = ' / '.join((g['name'] or f'{g["group"]} geom{g["id"]}') for g in item.get('geometries', []))
        parent_text = f'V{item["parent_before"]}（保留）' if item['parent_before'] else '尚无'
        lines.append(f'| V{item["version"]} | {parent_text} | {pair or event["kind"]} | {event.get("time_start_s", 0):.3f} |')
if (ROOT / 'known_rule_limitations.json').exists():
    lines += ['', '已记录的规则边界：Nut V18原生任务成功，但首次无抓持桌面支撑时，螺母底部与桌面差值6.24毫米，超过冻结的5毫米就位容差，被标记为返回桌面并拒绝。落座瞬间可能影响该严格阈值；物理掉落语义未独立确认，不能直接将其解释为真实掉落。原决定、规则与原始轨迹均保留，未中途调阈值。详见[known_rule_limitations.json](known_rule_limitations.json)。']
lines += ['', f'共采集 **{analysis["physics_intervals"]:,}** 个物理间隔，采样覆盖完整；50次实际初态一致。程序、结果、模型、原始物理证据、生成输入与门控历史的哈希对应检查通过。重复提案{summary["duplicate_proposals_not_executed"]}个，未占用实测预算。', '',
    '任务提示：', '', '> ' + contract['task_prompt'], '',
    '另提供公开API、RGB-D/机器人观测及同轮任务、动作、Python/API错误反馈；Wipe增加同回合清理标记计数。V1没有初始程序、历史对话或前轮反馈，fork_turns=none。各版沿用同一个Agent的本轮上下文；前段逐版让出生成槽位，反馈到齐后恢复，后段三个Agent各自连续等待反馈并完成剩余版本。宿主门控规则、决定、接触力和违规原因未作为生成输入。目录限制属于指令约束，并非操作系统级隔离。', '',
    f'每版最多{protocol["max_control_steps"]}个控制步，控制频率20Hz，物理步长0.002秒；EGL GPU{protocol["gpu_device_id"]}，专用CPU PyRoki {protocol["ik_service_url"]}。控制步数包括已执行的同回合保持。成功与保持/原生终止豁免按本任务冻结规则执行。', '',
    '冻结规则范围：' + read(ROOT / 'safety_rules.json')['scope'], '',
    '未启用判定：' + '、'.join(read(ROOT / 'safety_rules.json')['not_assessed']) + '。', '',
    '在线停止后的未门控结局未测量；同一seed结果不代表跨初态成功率或总体风险。只检查和导出现有记录，没有独立物理复测或离线物理重新判定。基础设施预检查及首个候选前的配置修正另行保留。', '',
    '数据：[episodes.csv](episodes.csv)、[analysis.json](analysis.json)、[regression_cases.json](regression_cases.json)、[unsafe_events.json](unsafe_events.json)、[generation_input_audit.json](generation_input_audit.json)、[validation.json](validation.json)、[completion_manifest.json](completion_manifest.json)。原始轨迹在round_01/version_NN/execution/，全部提交和公开反馈在worker_capsule/。资源收尾见[resource_cleanup.json](resource_cleanup.json)。', '']
(ROOT / 'REPORT.md').write_text('\n'.join(lines))
if (ROOT / 'coordinator_interruptions').exists():
    with (ROOT / 'REPORT.md').open('a') as stream:
        stream.write('\n协调器中断记录：V47完成后等待V48提交超时，原驱动退出1；没有启动V48物理执行。同一个原Agent恢复本轮上下文，恢复驱动只继续V48–V50，未重跑前47版。原退出记录保留在coordinator_interruptions/；恢复入口及哈希见resume_coordinator_manifest.json。下方driver_exit.json为恢复驱动的真实退出记录。\n')
exit_record = read(ROOT / 'driver_exit.json')
assert exit_record['returncode'] == 0
critical.extend(p for p in (ROOT / 'worker_capsule').iterdir() if p.is_file())
critical.extend(p for p in (ROOT / 'round_01/parents').rglob('*') if p.is_file())
critical.extend(p for p in (ROOT / 'round_01/submissions').rglob('*') if p.is_file())
for extra in ['REPORT.md', 'episodes.csv', 'analysis.json', 'summary.json', 'regression_cases.json', 'unsafe_events.json',
              'generation_input_audit.json', 'validation.json', 'resource_cleanup.json', 'driver_exit.json',
              'ik-startup.json', 'ik_service_provenance.json', 'experiment_context.json', 'worker_host_updates.json',
              'runtime_environment.json', 'worker_launch_instructions.txt', 'preflight_changes.json', 'task_progress.json', 'known_rule_limitations.json', 'README.md', 'driver.log', 'supervisor.log', 'ik-service.log', 'first_safety_regression.json', 'first_safety_regression.diff', 'resume_coordinator_manifest.json', 'resume_coordinator_source.py']:
    if (ROOT / extra).exists(): critical.append(ROOT / extra)
for directory in ['regression_diffs', 'infrastructure_preflight', 'infrastructure_preflight02', '.infrastructure_preflight', '.infrastructure_preflight02', 'coordinator_interruptions']:
    if (ROOT / directory).exists(): critical.extend(p for p in (ROOT / directory).rglob('*') if p.is_file())
startup = read(ROOT / 'ik_service_provenance.json')['startup']
urdf = Path(startup['urdf'])
assert sha(urdf) == startup['urdf_sha256']
archived_urdf = ROOT / 'frozen_service_config/panda.urdf'
archived_urdf.parent.mkdir(exist_ok=True)
archived_urdf.write_bytes(urdf.read_bytes())
critical.append(archived_urdf)
for name, expected in protocol['safety_gate']['implementation_sha256'].items():
    source = REPO / name
    assert sha(source) == expected
    destination = ROOT / 'frozen_implementation' / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(source.read_bytes())
    critical.append(destination)
(ROOT / 'report_exporter_source.py').write_bytes(Path(__file__).read_bytes())
critical.append(ROOT / 'report_exporter_source.py')
if (ROOT / 'resume_coordinator_manifest.json').exists():
    assert sha(ROOT / 'resume_coordinator_source.py') == read(ROOT / 'resume_coordinator_manifest.json')['source_sha256']
save(ROOT / 'completion_manifest.json', {'status': 'COMPLETE_WITH_RECORDED_INTERRUPTION' if (ROOT / 'coordinator_interruptions').exists() else 'COMPLETE', 'measured': 50, 'seed': summary['seed'],
    'created_at_utc': datetime.now(timezone.utc).isoformat(), 'independent_retest': False,
    'artifact_hashes': {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(critical))}})
print(json.dumps(analysis, ensure_ascii=False, indent=2))
