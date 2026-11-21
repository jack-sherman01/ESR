"""Evidence links, source diffs, raw witness checks and continuous measurements."""
from collections import defaultdict
import csv
import difflib
import json
from pathlib import Path

from esrb.runtime.io import digest, save
from .natural_study import matching_events, predicate_changes, require
from .natural_study_notes import REVIEWS


def verify_witness(execution, event, inputs):
    """Check the recorded contact row against the original compressed physics chunk."""
    import numpy as np
    contact = event.get('contact', event)
    if 'contact_row' not in contact:
        return {'status': 'NO_CONTACT_ROW', 'kind': event['kind'], 'note': 'State-event witness; no invented contact sample'}
    path = inputs.bind(Path(execution) / 'physics' / contact['chunk'])
    geometry = inputs.json(Path(execution) / 'physics/geometry.json')
    schema = inputs.json(Path(execution) / 'physics/schema.json')
    require(schema['physics_timestep_s'] == 0.002, 'Unexpected physics timestep')
    require([geometry[i]['geom_name'] for i in contact['geom_ids']] == contact['geom_names'], 'Witness geometry names mismatch')
    require([geometry[i]['body_name'] for i in contact['geom_ids']] == contact['body_names'], 'Witness body names mismatch')
    with np.load(path, allow_pickle=False) as chunk:
        row = chunk['contacts'][contact['contact_row']]
        local_index = int(np.searchsorted(chunk['contact_offsets'], contact['contact_row'], side='right')-1)
        np.testing.assert_allclose(chunk['time'][local_index], [contact['time_start_s'], contact['time_end_s']], rtol=0, atol=1e-8)
        require(row[:2].astype(int).tolist() == contact['geom_ids'], 'Witness geometry ids mismatch')
        np.testing.assert_allclose(row[3:6], contact['point_world_m'], rtol=0, atol=1e-12)
        np.testing.assert_allclose(row[15:21], contact['wrench_contact_raw'], rtol=0, atol=1e-12)
        np.testing.assert_allclose(row[6:15].reshape(3, 3).T @ row[15:18],
                                   contact['force_world_on_geom2_n'], rtol=1e-12, atol=1e-12)
    return {'status': 'PASS', 'kind': event['kind'], 'chunk': str(path), 'sha256': digest(path),
            'contact_row': contact['contact_row'], 'geom_names': contact['geom_names'],
            'physics_index': contact.get('physics_index'), 'time_start_s': contact.get('time_start_s'),
            'normal_force_n_first_witness': contact['normal_force_n']}


def api_timeline(native, inputs):
    path = Path(native.get('native_run', '')) / 'trace/events.jsonl'
    if not path.exists():
        return []
    inputs.bind(path)
    starts, rows = {}, []
    for line in path.read_text().splitlines():
        event = json.loads(line)
        if event.get('event') == 'span_start':
            starts[event['span_id']] = event
        elif event.get('event') == 'span_end' and event['span_id'] in starts:
            old = starts.pop(event['span_id'])
            rows.append({'name': old['name'], 'start_s': old.get('simulator_time_s'),
                         'end_s': event.get('simulator_time_s'), 'span_id': event['span_id'], 'parent_id': old.get('parent_id')})
    return rows


def build_cases(chains, lookup, native, tables, out, inputs):
    chain_lookup = {(c['task'], c['chain']): c for c in chains}
    selected = [(u, 'new_violation') for u in tables['updates'] if u['new_violation']]
    # Reproducible controls: earliest accepted persistent and resolved update per task.
    for task in sorted({c['task'] for c in chains}):
        for label, field in [('persistent_control', 'persistent_predicates'), ('repair_control', 'resolved_predicates')]:
            options = [u for u in tables['updates'] if u['task'] == task and u['task_accepted'] and u[field] and not u['new_violation']]
            if options:
                selected.append((min(options, key=lambda u: (u['chain'], u['version'])), label))
    cases, checks, generations = [], [], []
    for chain in chains:
        root = Path(chain['root'])
        for event in chain['events']:
            folder = root / f"candidates/v{event['version']:02d}"
            for attempt in sorted(folder.glob('generation_*')):
                if not attempt.is_dir():
                    continue
                cp = attempt / 'generation_config.json'; rp = attempt / 'model_receipt.json'
                config = inputs.json(cp) if cp.exists() else {}
                receipt = inputs.json(rp) if rp.exists() else {}
                for filename, field in [('prompt.txt', 'prompt_sha256'), ('response.json', 'response_sha256')]:
                    if (attempt / filename).exists():
                        inputs.bind(attempt / filename, receipt.get(field))
                usages = [c.get('usage', {}) for c in receipt.get('completion', [])]
                generations.append({'task': chain['task'], 'chain': chain['chain'], 'version': event['version'],
                    'attempt': attempt.name, 'requested_model': receipt.get('requested_model', config.get('model')),
                    'automatic_fallback': receipt.get('automatic_fallback'),
                    'reasoning_effort': config.get('reasoning_effort'), 'history_mode': config.get('history'),
                    'prompt_chars': config.get('prompt_chars'), 'usage_recorded': bool(usages),
                    'input_tokens': sum(u.get('input_tokens', 0) for u in usages) if usages else None,
                    'output_tokens': sum(u.get('output_tokens', 0) for u in usages) if usages else None,
                    'cached_input_tokens': sum(u.get('cached_input_tokens', 0) for u in usages) if usages else None})
    tables['generation_usage'] = generations
    evidence_dir = out / 'cases'; evidence_dir.mkdir()
    for update, selection in selected:
        task, chain, version, parent = [update[k] for k in ['task', 'chain', 'version', 'parent_version']]
        info = chain_lookup[(task, chain)]; root = Path(info['root'])
        folder = root / f'candidates/v{version:02d}'
        proposal = inputs.json(folder / 'proposal.json')
        oldp = root / f'candidates/v{parent:02d}/program.py'; newp = folder / 'program.py'
        old, new = inputs.text(oldp), inputs.text(newp)
        changes = list(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                                           fromfile=str(oldp), tofile=str(newp)))
        name = f'{task}-c{chain}-v{parent:02d}-v{version:02d}-{selection}'
        diff_path = evidence_dir / (name + '.diff'); diff_path.write_text(''.join(changes))
        witnesses, results = [], []
        for seed in info['seeds']:
            key = (task, chain, version, seed); oldkey = (task, chain, parent, seed)
            before, after = lookup[oldkey], lookup[key]
            predicates = predicate_changes(before, after)
            relevant = predicates['new'] if selection == 'new_violation' else predicates['persistent'] if selection == 'persistent_control' else predicates['resolved']
            result = before if selection == 'repair_control' else after
            trace = api_timeline(native[key], inputs)
            old_trace = api_timeline(native[oldkey], inputs)
            events = matching_events(result, relevant)
            for event in events:
                check = verify_witness(result['execution'], event, inputs)
                checks.append(dict(case=name, seed=seed, **check))
                t = event.get('time_start_s', event.get('time_s'))
                context = [span for span in (old_trace if selection == 'repair_control' else trace)
                           if t is not None and span['start_s'] is not None and span['end_s'] is not None
                           and span['start_s']-0.002 <= t <= span['end_s']+0.002]
                witnesses.append({'seed': seed, 'source_version': parent if selection == 'repair_control' else version,
                    'predicates': relevant, 'event': event, 'raw_check': check, 'api_spans_at_witness': context})
            results.append({'seed': seed, 'initial_state_id': after['initial_state_id'], 'model_sha256': after['model_sha256'],
                'parent_task_qualified': before['task_qualified'], 'child_task_qualified': after['task_qualified'],
                'parent_audit': before['audit_status'], 'child_audit': after['audit_status'],
                'parent_control_steps': before['control_steps'], 'child_control_steps': after['control_steps'],
                'parent_execution': before['execution'], 'child_execution': after['execution'],
                'child_native_run': native[key].get('native_run'), 'parent_api_timeline': old_trace,
                'child_api_timeline': trace, 'child_phase_segments': after.get('phase_segments', [])})
        require(selection != 'new_violation' or witnesses, 'New violation lacks a matching physical witness')
        # Categories describe literal changed code tokens; never a causal attribution.
        edited = ''.join(line[1:] for line in changes if line[:1] in {'+', '-'} and line[:3] not in {'+++', '---'})
        token_groups = {'pose_or_path_tokens': ['pose', 'height', 'offset', 'quat', 'position'],
                        'gripper_tokens': ['gripper', 'grasp', 'release'],
                        'retry_or_control_flow_tokens': ['retry', 'attempt', 'for ', 'while ', 'if '],
                        'perception_tokens': ['observation', 'locate', 'mask', 'point_cloud']}
        tags = [tag for tag, tokens in token_groups.items() if any(token in edited.lower() for token in tokens)]
        case = {'id': name, 'selection': selection, **update, 'rationale': proposal.get('rationale'),
                'parent_program': str(oldp), 'child_program': str(newp),
                'parent_program_sha256': digest(oldp), 'child_program_sha256': digest(newp),
                'diff': str(diff_path.relative_to(out)), 'literal_changed_token_tags': tags,
                'interpretation': '源码差异与物理事件的观察关联；Agent 理由不是因果证据。',
                'five_seed_results': results, 'witnesses': witnesses}
        note = REVIEWS.get((task, chain, version))
        if selection == 'new_violation' and note and note['parent_program_sha256'] == case['parent_program_sha256'] and note['child_program_sha256'] == case['child_program_sha256']:
            case['source_review'] = note['source_review']
            case['causal_status'] = note['causal_status']
        else:
            case['source_review'] = '未使用预存源码复核说明；以附带 diff 和物理证据为准。'
        cases.append(case)
        lines = [f'# {name}', '', f'选择：{selection}；任务接纳：{update["task_accepted"]}；严格父面板 PASS：{update["all_five_parent_pass"]}。',
                 '', '## Agent 原始修改理由', '', proposal.get('rationale') or '未记录', '',
                 '## 程序变化', '', f'[完整 diff]({diff_path.name})', '', '```diff', ''.join(changes).rstrip(), '```', '',
                 '## 五 seed 配对结果', '', '| seed | 父/子任务合格 | 父/子审计 | 父/子控制步 |', '|---|---|---|---|']
        for r in results:
            lines.append(f'| {r["seed"]} | {r["parent_task_qualified"]}/{r["child_task_qualified"]} | {r["parent_audit"]}/{r["child_audit"]} | {r["parent_control_steps"]}/{r["child_control_steps"]} |')
        lines += ['', '## 物理证据', '', '以下接触力是首个证据采样的法向力，不是整回合峰值，也不代表损伤等级。', '']
        for w in witnesses:
            c = w['event'].get('contact', w['event'])
            lines += [f'- seed {w["seed"]}，V{w["source_version"]}，规则 {w["event"]["kind"]}：'
                      f'时间 {c.get("time_start_s", w["event"].get("time_s"))} s；'
                      f'部件 {c.get("geom_names", [])}；法向力 {c.get("normal_force_n")} N；'
                      f'原始核对 {w["raw_check"]["status"]}。']
        lines += ['', '完整初态标识、执行目录、阶段与 API 时间线见上级 cases.json。', '', case['interpretation'], '']
        lines += ['## 源码复核说明', '', case['source_review'], '', '该说明是事后源码审阅，不是 Agent 当时获得的反馈，也不是因果实验。', '']
        (evidence_dir / (name + '.md')).write_text('\n'.join(lines), encoding='utf-8')
    return cases, checks


def continuous_tables(audit, lookup, inputs):
    """Reuse frozen numeric measurements; invalidate comparisons without dynamic evidence."""
    changes = []
    p = audit / 'continuous_changes.csv'; inputs.bind(p)
    with p.open() as stream:
        for row in csv.DictReader(stream):
            key = (row['task'], int(row['chain']), int(row['version']), int(row['seed']))
            oldkey = (row['task'], int(row['chain']), int(row['parent_version']), int(row['seed']))
            old, new = lookup[oldkey], lookup[key]
            dynamic = all(r.get('metrics', {}).get('physics_steps', 0) > 0 and r['coverage_complete'] for r in [old, new])
            if not dynamic:
                row.update(status='NOT_COMPARABLE', delta_child_minus_parent='')
            row['interpretation'] = 'continuous measurement only; no new threshold or overall safety verdict'
            changes.append(row)
    body = {}
    p = audit / 'body_interface_loads.csv'; inputs.bind(p)
    with p.open() as stream:
        for row in csv.DictReader(stream):
            key = (row['id'], row['interface'].split('|phase=')[0])
            fields = ['peak_force_magnitude_sum_n', 'peak_net_force_n', 'peak_body_origin_moment_nm', 'active_s']
            if key not in body:
                body[key] = {field: float(row[field]) for field in fields}
            else:
                for field in fields:
                    body[key][field] = (body[key][field]+float(row[field]) if field == 'active_s' else
                                        max(body[key][field], float(row[field])))
    # Compact summaries by task/interface/measurement, preserving units and sample counts.
    groups = defaultdict(list)
    for row in changes:
        if row['status'] == 'MEASURED' and row['delta_child_minus_parent'] != '':
            groups[(row['task'], row['interface'], row['channel'], row['measurement'], row['unit'])].append(float(row['delta_child_minus_parent']))
    summary = []
    for key, values in sorted(groups.items()):
        summary.append(dict(zip(['task', 'interface', 'channel', 'measurement', 'unit'], key),
                            paired_cases=len(values), mean_delta=sum(values)/len(values),
                            min_delta=min(values), max_delta=max(values), positive_deltas=sum(x > 0 for x in values),
                            interpretation='descriptive continuous changes; correlated updates, no calibrated overload verdict'))
    bodygroups = defaultdict(list)
    body_interfaces = defaultdict(set)
    for (episode, interface), values in body.items():
        bodygroups[(episode.split('/')[0], interface)].append(values)
        body_interfaces[episode].add(interface)
    body_summary = []
    for (task, interface), values in sorted(bodygroups.items()):
        for field, unit in [('peak_force_magnitude_sum_n', 'N'), ('peak_net_force_n', 'N'),
                            ('peak_body_origin_moment_nm', 'Nm'), ('active_s', 's')]:
            samples = sorted(v[field] for v in values)
            n = len(samples)
            body_summary.append({'task': task, 'interface': interface, 'measurement': field, 'unit': unit,
                'episodes_with_interface_record': n, 'min': samples[0], 'median': (samples[(n-1)//2]+samples[n//2])/2,
                'max': samples[-1], 'interpretation': 'observed interface records only; no calibrated damage threshold'})
    moment_changes = []
    for key, child in sorted(lookup.items()):
        if child['parent_version'] is None:
            continue
        old = lookup[(key[0], key[1], child['parent_version'], key[3])]
        if not all(r['coverage_complete'] and r.get('metrics', {}).get('physics_steps', 0) > 0 for r in [old, child]):
            continue
        interfaces = body_interfaces[child['id']] | body_interfaces[old['id']]
        for interface in sorted(interfaces):
            before = body.get((old['id'], interface), {}).get('peak_body_origin_moment_nm', 0.0)
            after = body.get((child['id'], interface), {}).get('peak_body_origin_moment_nm', 0.0)
            moment_changes.append({'task': child['task'], 'chain': child['chain'], 'version': child['version'],
                'parent_version': child['parent_version'], 'seed': child['seed'], 'task_accepted': child['task_accepted'],
                'interface': interface, 'parent_peak_moment_nm': before, 'child_peak_moment_nm': after,
                'delta_peak_moment_nm': after-before, 'moment_origin': 'measured body origin, not COM',
                'interpretation': 'complete dynamic traces; absent contact interface = zero; no torque violation verdict'})
    return {'continuous_changes': changes, 'continuous_summary': summary,
            'body_interface_summary': body_summary, 'paired_body_moments': moment_changes}
