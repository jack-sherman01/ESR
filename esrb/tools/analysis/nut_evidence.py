"""Explain Nut audit evidence in a new output directory without raw replay."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import csv
import json
import shutil

from esrb.detection.nut_evidence import diagnose
from esrb.runtime.io import digest, read, save
from esrb.runtime.paths import project_root
from .natural_audit import result_path, verify_freeze


def explain(previous, root, profile):
    previous, root, profile = Path(previous), Path(root), Path(profile)
    verify_freeze(previous)
    if read(previous / 'summary.json')['status'] != 'COMPLETE':
        raise ValueError('source audit incomplete')
    definitions = read(profile)
    if digest(previous / 'rules.json') != definitions['base_rules_sha256']:
        raise ValueError('definitions do not match frozen physical rules')
    manifest = previous / 'consolidated-artifacts.json'
    artifacts = read(manifest if manifest.exists() else previous / 'artifacts.json')
    items = [r for r in read(previous / 'inputs.json')['episodes'] if r['task'] == 'nut-assembly']
    if len({r['id'] for r in items}) != len(items):
        raise ValueError('duplicate source episodes')
    # Freeze the supplement before diagnosis. The original raw validation is
    # inherited from v2, not falsely advertised as a new full NPZ scan.
    root.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(profile, root / 'definitions.json')
    sources = {}
    for name in ['esrb/detection/nut_evidence.py', 'esrb/tools/analysis/nut_evidence.py']:
        dest = root / 'frozen_sources' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(project_root() / name, dest)
        sources[name] = digest(dest)
    save(root / 'protocol.json', {
        'schema': 'esrb.nut_evidence_report/1',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'previous': str(previous.resolve()),
        'previous_protocol_sha256': digest(previous / 'protocol.json'),
        'previous_artifacts_sha256': digest(manifest if manifest.exists() else previous / 'artifacts.json'),
        'definitions_sha256': digest(root / 'definitions.json'), 'sources': sources,
        'raw_physics_validation': 'Inherited hash-bound v2 validation; this report rechecks metadata, not every NPZ',
        'new_physics_episodes': 0, 'new_model_calls': 0,
        'scope': 'Post-hoc evidence classification; original physical verdicts and task lineage retained',
    })
    rows, bindings = [], []
    for item in items:
        source = result_path(previous, item)
        if digest(source) != artifacts[str(source.relative_to(previous))]:
            raise ValueError('source audit artifact mismatch: ' + item['id'])
        audit = read(source)
        execution = Path(item['execution'])
        raw_artifacts = read(execution / 'artifacts.json')
        files = [execution / 'result.json', execution / 'physics/summary.json']
        result, capture = read(files[0]), read(files[1])
        for key in ['id', 'execution', 'result_sha256', 'program_sha256', 'initial_state_id', 'model_sha256']:
            if audit.get(key) != item.get(key):
                raise ValueError('audit identity mismatch: ' + key)
        if digest(files[0]) != item['result_sha256']:
            raise ValueError('result identity mismatch')
        for key in ['program_sha256', 'initial_state_id', 'seed']:
            if result.get(key) != item.get(key):
                raise ValueError('result lineage mismatch: ' + key)
        program = None
        # Only zero-execution episodes need native program diagnostics. Read
        # the archived data, not the current API or current model services.
        if audit.get('metrics', {}).get('physics_steps') == 0:
            program_path = Path(result['native_run']) / 'source/program-result.json'
            if not program_path.is_relative_to(execution):
                raise ValueError('native diagnostic outside execution directory')
            files.append(program_path)
            program = read(program_path)
        hashes = {}
        for path in files:
            sha = digest(path)
            if sha != raw_artifacts[str(path.relative_to(execution))]:
                raise ValueError('raw metadata artifact mismatch: ' + str(path))
            hashes[str(path)] = sha
        row = diagnose(audit, result, capture, definitions, program)
        row.update(source_audit=str(source.resolve()), source_audit_sha256=digest(source),
                   model_sha256=item['model_sha256'], initial_state_id=item['initial_state_id'],
                   program_sha256=item['program_sha256'])
        rows.append(row)
        bindings.append({'id': item['id'], 'audit_sha256': row['source_audit_sha256'],
                         'metadata_sha256': hashes,
                         'program_error': (program or {}).get('error'),
                         'failed_calls': [c for c in (program or {}).get('calls', []) if c.get('ok') is False]})
    for chain in read(previous / 'inputs.json')['chains']:
        if chain['task'] == 'nut-assembly' and digest(Path(chain['root']) / 'state.json') != chain['state_sha256']:
            raise ValueError('source task lineage changed')
    summary = {
        'status': 'COMPLETE', 'episodes': len(rows),
        'legacy_audit_status': dict(Counter(r['legacy_audit_status'] for r in rows)),
        'execution_evidence_status': dict(Counter(r['execution_evidence_status'] for r in rows)),
        'no_execution_reasons': dict(Counter(r['execution_reason'] for r in rows
                                            if r['execution_evidence_status'] == 'NOT_EVALUATED_NO_EXECUTION')),
        'assessed_verdict': dict(Counter(r['assessed_verdict'] for r in rows)),
        'drop_precondition': dict(Counter(r['drop_precondition'] for r in rows)),
        'stable_seated_release': dict(Counter(r['stable_seated_release'] for r in rows)),
        'unknown_reason_episode_counts': dict(Counter(s for r in rows for s in r['unknown_reasons'])),
        'new_physics_episodes': 0, 'new_model_calls': 0,
        'legacy_results_unchanged': True, 'original_nut_task_lineages_unchanged': True,
        'limitations': [
            'No new physical rule labels; adds execution/evidence definitions to frozen v2',
            'No execution is not missing samples and is never safety PASS',
            'Complete sampling does not imply every risk is decidable',
            'Qualified overload, torque and jam thresholds remain unavailable',
            'Stable seating is not inferred from native success or terminal geometry alone',
        ],
    }
    save(root / 'summary.json', summary)
    save(root / 'episodes.json', rows)
    save(root / 'bindings.json', bindings)
    with (root / 'episodes.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                         for k, v in row.items()} for row in rows)
    lines = ['# NutAssemblySquare：UNKNOWN 与装配证据补充', '',
             '本报告仅补充证据定义；不改写冻结 v2 判定、不重跑仿真、不改变任务谱系。', '',
             '|旧版判定|回合数|', '|---|---:|']
    lines += [f'|{k}|{v}|' for k, v in summary['legacy_audit_status'].items()]
    lines += ['', '|执行证据状态|回合数|', '|---|---:|']
    lines += [f'|{k}|{v}|' for k, v in summary['execution_evidence_status'].items()]
    lines += ['', '|零执行原因|回合数|', '|---|---:|']
    lines += [f'|{k}|{v}|' for k, v in summary['no_execution_reasons'].items()]
    lines += ['', '无动作回合不进入可判定安全分母，仍计入全部执行尝试和任务结果；不能改填 PASS。', '',
              '一个已检出违规的回合仍有未确认失持时，保留违规及未确认项，不将未确认项伪造成第二个违规。', '',
              '部件映射、合法接触、插入走廊、掉落/崩落、稳定释放及卡滞资格定义见 definitions.json；继承冻结 v2 物理规则。', '',
              '过载、力矩和合格卡滞仍缺可迁移阈值；本补充没有凭事件数拟合新阈值。']
    (root / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    save(root / 'artifacts.json', {str(p.relative_to(root)): digest(p)
                                  for p in sorted(root.rglob('*')) if p.is_file()})
    return summary


def main(args):
    print(json.dumps(explain(args.previous, args.root, args.profile), ensure_ascii=False, indent=2))
    return 0
