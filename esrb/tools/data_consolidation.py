"""Consolidate completed evidence, preserving bytes before removing revisions.

Current files take precedence. Superseded execution receipts and unique attempts
are addressable through a hash index inside the surviving dataset, rather than
through separate repair folders. Never change raw samples or infer safety from
an explained execution failure.
"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import csv
import fcntl
import hashlib
import json
import os
import shutil
import time

from esrb.runtime.io import read, save, digest
from esrb.runtime.paths import project_root


def now():
    return datetime.now(timezone.utc).isoformat()


def entries(root):
    """Do not traverse external source/runtime symlinks."""
    root = Path(root)
    if root.is_file() or root.is_symlink():
        yield root
        return
    for directory, dirs, files in os.walk(root, followlinks=False):
        for name in list(dirs):
            p = Path(directory) / name
            if p.is_symlink():
                dirs.remove(name)
                yield p
        for name in files:
            yield Path(directory) / name


def link_copy(source, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    if source.is_symlink():
        dest.symlink_to(os.readlink(source), target_is_directory=source.is_dir())
    else:
        try:
            os.link(source, dest)
        except OSError:
            shutil.copy2(source, dest)


def overlay(source, target, exclude=()):
    for p in entries(source):
        relative = p.relative_to(source)
        if relative.parts[0] not in exclude:
            link_copy(p, target / relative)


def csv_save(path, rows, fields=None):
    path = Path(path)
    temporary = path.with_name(path.name + '.merge-tmp')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]), extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def protect_sources(stage, roots):
    """Index every absorbed file and retain each distinct byte sequence once.

    A source not present in the current dataset is stored by SHA-256. This keeps
    necessary failed-attempt evidence without retaining old revision trees.
    Hashing uses the inode cache only within this locked, completed transaction.
    """
    provenance = stage / 'provenance'
    provenance.mkdir(exist_ok=True)
    shutil.copyfile(Path(__file__), provenance / 'consolidator.py')
    cache, locations, rows = {}, {}, []

    def identity(path):
        stat = path.stat()
        key = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        if key not in cache:
            cache[key] = digest(path)
        return cache[key]

    for p in entries(stage):
        if not p.is_symlink():
            locations.setdefault(identity(p), str(p.relative_to(stage)))
    for root in roots:
        for p in entries(root):
            row = {'source': str(p), 'source_root': str(root), 'relative': str(p.relative_to(root))}
            if p.is_symlink():
                row.update(kind='symlink', target=os.readlink(p))
            else:
                value = identity(p)
                if value not in locations:
                    dest = provenance / 'objects' / value[:2] / value
                    link_copy(p, dest)
                    locations[value] = str(dest.relative_to(stage))
                row.update(kind='file', sha256=value, size=p.stat().st_size, stored=locations[value])
            rows.append(row)
    save(provenance / 'source-index.json', rows)
    # Independently hash the retained bytes, not merely trust the copy operation.
    with ThreadPoolExecutor(max_workers=4) as pool:
        checks = list(pool.map(lambda item: digest(stage / item[1]) == item[0], locations.items()))
    if not all(checks):
        raise ValueError('retained content hash mismatch')
    save(provenance / 'absorption.json', {
        'status': 'PASS', 'at_utc': now(), 'source_roots': [str(p) for p in roots],
        'source_entries': len(rows), 'distinct_file_contents_verified': len(locations),
        'consolidator_sha256': digest(provenance / 'consolidator.py'),
        'policy': 'Current view uses latest records; hash index retains original execution receipts and distinct attempts',
    })
    return rows


def tree_manifest(root):
    return {str(p.relative_to(root)): {'symlink': os.readlink(p)} if p.is_symlink() else digest(p)
            for p in entries(root) if p.name != 'consolidated-artifacts.json'}


def publish(stage, target, absorbed):
    """Publish, verify byte identity, then remove only fully absorbed roots."""
    backup = target.with_name('.' + target.name + '.pre-merge')
    if backup.exists():
        raise ValueError('unfinished publication requires recovery: ' + str(backup))
    manifest = tree_manifest(stage)
    save(stage / 'consolidated-artifacts.json', manifest)
    target.rename(backup)
    try:
        stage.rename(target)
        if tree_manifest(target) != manifest:
            raise ValueError('published data differs from verified stage')
    except BaseException:
        if target.exists():
            target.rename(stage)
        backup.rename(target)
        raise
    # All source bytes were absorbed and independently verified before rename.
    # Source trees other than the target must not have been changed in between.
    source_index = read(target / 'provenance/source-index.json')
    by_root = {}
    for row in source_index:
        by_root.setdefault(row['source_root'], []).append(row)
    for root in absorbed:
        actual = backup if root == target else root
        rows = by_root[str(root)]
        actual_entries = {str(p.relative_to(actual)) for p in entries(actual)}
        if actual_entries != {r['relative'] for r in rows}:
            raise ValueError('source file set changed before cleanup: ' + str(root))
        for row in rows:
            p = actual / row['relative']
            if row['kind'] == 'file':
                stored = target / row['stored']
                if digest(p) != row['sha256'] or digest(stored) != row['sha256']:
                    raise ValueError('source changed before cleanup: ' + str(p))
            elif not p.is_symlink() or os.readlink(p) != row['target']:
                raise ValueError('source symlink changed before cleanup')
    deleted = []
    for root in absorbed:
        actual = backup if root == target else root
        if actual.is_dir() and not actual.is_symlink():
            shutil.rmtree(actual)
        else:
            actual.unlink()
        deleted.append(str(root))
    save(target / 'cleanup.json', {'status': 'COMPLETE', 'at_utc': now(),
        'removed_absorbed_source_trees': deleted,
        'note': 'Target name retained with current merged data; required source bytes are indexed within provenance'})
    save(target / 'consolidated-artifacts.json', tree_manifest(target))


def resolve_original(root, original):
    """Find an archived execution receipt without recreating deleted folders."""
    root, original = Path(root), str(original)
    for row in read(root / 'provenance/source-index.json'):
        if row['source'] == original:
            if row['kind'] != 'file':
                return row['target']
            return str(root / row['stored'])
    raise FileNotFoundError(original)


def natural(repo, status):
    from esrb.detection.nut_evidence import execution_evidence
    base = (repo / 'private/analysis').resolve()
    target = base / 'natural-safety-20261119'
    extended = base / 'natural-safety-extended-v2-20261119'
    nut = base / 'nut-evidence-definitions-20261120'
    if (target / 'cleanup.json').exists() and not extended.exists() and not nut.exists():
        return read(target / 'summary.json')
    if read(extended / 'summary.json')['status'] != 'COMPLETE' or read(nut / 'summary.json')['status'] != 'COMPLETE':
        raise ValueError('natural analysis inputs incomplete')
    stage = base / '.natural-consolidation-staging'
    stage.mkdir(exist_ok=False)
    overlay(extended, stage)
    # This diagnostic report belongs to the same dataset, not another revision.
    overlay(nut, stage / 'diagnostics/nut')
    definitions = read(nut / 'definitions.json')
    inputs = read(extended / 'inputs.json')
    nut_rows = {r['id']: r for r in read(nut / 'episodes.json')}
    rows, audits, changes = [], [], []
    for item in inputs['episodes']:
        relative = Path('episodes') / item['task'] / f"chain_{item['chain']:02d}" / f"v{item['version']:02d}_seed_{item['seed']}.json"
        audit = read(extended / relative)
        old = read(target / relative)
        if old['audit_status'] == 'VIOLATION' and audit['audit_status'] != 'VIOLATION':
            raise ValueError('previously witnessed violation lost')
        result = read(Path(item['execution']) / 'result.json')
        if digest(Path(item['execution']) / 'result.json') != item['result_sha256']:
            raise ValueError('original natural result changed')
        capture = read(Path(item['execution']) / 'physics/summary.json')
        program = None
        if audit.get('metrics', {}).get('physics_steps') == 0:
            program = read(Path(result['native_run']) / 'source/program-result.json')
        evidence, reason = execution_evidence(audit, result, capture, definitions, program)
        audit['execution_evidence_status'] = evidence
        audit['execution_reason'] = reason
        if program is not None:
            audit['execution_diagnostic'] = {
                'program_error': program.get('error'),
                'failed_calls': [c for c in program.get('calls', []) if c.get('ok') is False],
            }
        audit['previous_audit_status'] = audit['audit_status']
        if evidence == 'NOT_EVALUATED_NO_EXECUTION' and audit['audit_status'] == 'UNKNOWN':
            audit['audit_status'] = 'NOT_EVALUATED'
            audit['previous_unknown_reasons'] = sorted(set(audit['unknown']))
            audit['unknown'] = []
            changes.append({'id': item['id'], 'previous': 'UNKNOWN', 'current': 'NOT_EVALUATED', 'reason': reason})
        if item['id'] in nut_rows:
            audit['nut_evidence'] = {k: v for k, v in nut_rows[item['id']].items()
                                     if k not in ('source_audit', 'source_audit_sha256')}
        save(stage / relative, audit)
        audits.append(audit)
        rows.append({k: audit.get(k) for k in ['id','task','chain','version','seed','parent_version','task_accepted',
            'task_qualified','program_ok','native_task_success','audit_status','coverage_complete','control_steps']} |
            {'violation_kinds': ';'.join(sorted({e['kind'] for e in audit['events'] if e['kind'] != 'unconfirmed_drop'})),
             'unknown': ';'.join(audit['unknown']), 'physics_steps': audit['metrics'].get('physics_steps', 0),
             'execution_evidence_status': evidence, 'execution_reason': reason})
        # The former baseline reference is historical, not a live dependency on
        # a deleted revision. Its original byte identity remains in provenance.
        item['previous_audit_sha256'] = item.pop('base_audit_sha256', None)
        item.pop('base_audit_file', None)
    for chain in inputs['chains']:
        if digest(Path(chain['root']) / 'state.json') != chain['state_sha256']:
            raise ValueError('natural task lineage changed')
    csv_save(stage / 'episodes.csv', rows)
    summary = read(extended / 'summary.json')
    summary['episode_status'] = dict(Counter(a['audit_status'] for a in audits))
    summary['not_evaluated'] = sum(a['audit_status'] == 'NOT_EVALUATED' for a in audits)
    summary['consolidation'] = {'at_utc': now(), 'status': 'COMPLETE', 'reason_changes': changes,
        'old_versions_replaced': True, 'scope': 'No execution is not physical safety PASS; raw data and task lineage unchanged'}
    for scene in summary['scenes']:
        own = [a for a in audits if a['task'] == scene['task']]
        scene['unknown'] = sum(a['audit_status'] == 'UNKNOWN' for a in own)
        scene['not_evaluated'] = sum(a['audit_status'] == 'NOT_EVALUATED' for a in own)
    for chain in summary['chains']:
        own = [a for a in audits if a['task'] == chain['task'] and a['chain'] == chain['chain']]
        chain['unknown'] = sum(a['audit_status'] == 'UNKNOWN' for a in own)
        chain['not_evaluated'] = sum(a['audit_status'] == 'NOT_EVALUATED' for a in own)
    save(stage / 'summary.json', summary)
    csv_save(stage / 'scenes.csv', summary['scenes'])
    csv_save(stage / 'chains.csv', summary['chains'])
    save(stage / 'inputs.json', inputs)
    protocol = read(extended / 'protocol.json')
    historical_base = {k: protocol.pop(k) for k in list(protocol) if k.startswith('base_')}
    protocol.update(schema='esrb.consolidated_natural_audit/1', inputs_sha256=digest(stage / 'inputs.json'),
                    consolidation_at_utc=now(), original_natural_data_unchanged=True,
                    historical_base_receipt=historical_base)
    save(stage / 'protocol.json', protocol)
    # Old manifests describe old bytes. Keep them as indexed execution receipts,
    # not as an apparently valid manifest for the newly consolidated view.
    for name in ['artifacts.json', 'verification.json']:
        (stage / name).unlink(missing_ok=True)
    report = ['# 自然更新：合并后的当前安全审计', '',
        '750 个候选，3750 个开发回合。扩展物理规则与明确的未执行原因已合并到各回合记录。', '',
        '|场景|规则 PASS|违规|未执行、不可评价|其他 UNKNOWN|', '|---|---:|---:|---:|---:|']
    report += [f"|{s['task']}|{s['assessed_pass']}|{s['violation']}|{s['not_evaluated']}|{s['unknown']}|" for s in summary['scenes']]
    report += ['', '未执行不等于安全通过；未校准的载荷、力矩及卡滞风险仍逐规则 UNKNOWN。', '',
               '旧审计和补充目录已吸收；原始执行回执、规则来源和不同尝试通过 provenance/source-index.json 按哈希查找。',
               '原始轨迹、Agent 输入及任务父子谱系未修改。更新退化数量仍为 21 个提案、其中 7 个被任务选择接纳。']
    (stage / 'REPORT.md').unlink(missing_ok=True)
    (stage / 'REPORT.md').write_text('\n'.join(report) + '\n')
    # Historical findings remain receipts, not a second "current" report.
    (stage / 'FINDINGS.md').unlink(missing_ok=True)
    (stage / 'FINDINGS.md').write_text('[当前合并报告](REPORT.md)\n')
    save(stage / 'verification.json', {'status': 'PASS', 'episodes': len(rows),
        'new_physics_episodes': 0, 'new_model_calls': 0, 'original_lineages_unchanged': True,
        'previous_witnessed_violations_preserved': True, 'reason_changes': changes})
    status.update(phase='NATURAL_HASH_VERIFY', updated_at_utc=now())
    protect_sources(stage, [target, extended, nut])
    publish(stage, target, [target, extended, nut])
    return summary


def controlled_ready(base):
    path = base / 'controlled-20261120-r9-rules-v6'
    status = read(path / 'reaudit-status.json')
    if status['status'] == 'FAILED':
        raise ValueError('latest controlled re-audit failed; no cleanup: ' + str(status))
    if status['status'] != 'COMPLETE':
        return status
    receipt = project_root() / '.esr-local/job_specs/measurement-rules-v6-r9.status.json'
    if receipt.exists() and read(receipt).get('status') != 'FINISHED':
        return {'status': 'WAITING_FOR_COORDINATOR_EXIT'}
    if not (path / 'rule-revision-audit.json').exists() or not (path / 'report/summary.json').exists():
        return {'status': 'WAITING_FOR_FINAL_REPORT', **status}
    return None


def controlled(repo, status):
    base = (repo / '.esr-local/measurement-validation').resolve()
    target = base / 'controlled-20261120-r5'
    repair = base / 'controlled-20261120-r8-reprocess'
    latest = base / 'controlled-20261120-r9-rules-v6'
    stage = base / '.controlled-consolidation-staging'
    stage.mkdir(exist_ok=False)
    overlay(target, stage)
    # Keep the original physical source path: compiled models and raw requests
    # refer to this capsule. Later analysis capsules are bound separately.
    exclude = ('source', 'source_hashes.json', 'sources.tar.gz', 'runtime_cache')
    for revision in [repair, latest]:
        overlay(revision, stage, exclude=exclude)
        capsule = stage / 'analysis-capsules' / revision.name
        for name in ['source', 'source_hashes.json', 'sources.tar.gz', 'protocol.json']:
            p = revision / name
            if p.is_dir(): overlay(p, capsule / name)
            elif p.exists(): link_copy(p, capsule / name)
    # Repeated deterministic failures are attempts, not extra main samples.
    replay = base / 'controlled-20261120-r5-structural-replay-v5'
    replay_summary = read(replay / 'replay_comparison.json')
    if replay_summary['physical_attempts'] != 24 or replay_summary['reproduced_solver_time_resets'] != 24:
        raise ValueError('unexpected supplemental replay inventory')
    cases = replay_summary['cases']
    for case in cases:
        relative = Path(case['case'])
        if not case['physical_chunk_manifests_identical']:
            raise ValueError('structural replay changed physics; investigate before merge')
        old_manifest = read(stage / relative / 'raw/manifest.json')
        new_manifest = read(replay / relative / 'raw/manifest.json')
        if old_manifest['chunks'] != new_manifest['chunks']:
            raise ValueError('replay chunks differ from canonical case')
        row = read(stage / relative / 'result.json')
        row['repeat_execution_evidence'] = {
            'attempts': 1, 'same_initial_and_model': case['initial_match'] and case['model_match'],
            'identical_physics_chunks': True, 'reproduced_time_reset': True,
            'interpretation': 'Deterministic numerical failure; missing physics validity remains UNKNOWN',
        }
        save(stage / relative / 'result.json', row)
    protocol = read(latest / 'protocol.json')
    original = read(target / 'protocol.json')
    protocol.update(source_root=str(target / 'source'), source_sha256=original['source_sha256'],
        analysis_source_root=str(target / 'analysis-capsules' / latest.name / 'source'),
        analysis_source_manifest=str(target / 'analysis-capsules' / latest.name / 'source_hashes.json'),
        consolidation_at_utc=now(), canonical_dataset=str(target),
        consolidation_scope='Latest v6 analysis + completed repairs + original raw data; old request paths are historical receipts resolved by provenance/source-index.json')
    save(stage / 'protocol.json', protocol)
    all_rows = [read(p) for phase in ('calibration','validation','gate','replay')
                for p in (stage / phase).glob('*/*/result.json')]
    if len(all_rows) != 1524 or any(r.get('status') != 'RECORDED' for r in all_rows):
        raise ValueError('not all 1524 slots are recorded')
    save(stage / 'consolidation.json', {'status': 'COMPLETE', 'at_utc': now(),
        'main_episodes': 1524, 'supplemental_repeat_attempts': 24,
        'repaired_original_slots': 115, 'rules': 'measurement-event-profile/6',
        'physical_source_preserved': True, 'raw_physics_modified': False,
        'new_physics_episodes': 0, 'new_model_calls': 0,
        'phase_status': {phase: dict(Counter(r['status'] for r in all_rows if r.get('file','').startswith(phase+'/')))
                         for phase in ('calibration','validation','gate','replay')}})
    # Limit cleanup to the actual completed supplements explicitly requested
    # for merging. Failed revisions, loose logs and historical exports are not
    # deleted. In particular the original R7 execution capsule stays available.
    absorbed = [target, repair, latest, replay]
    status.update(phase='CONTROLLED_HASH_VERIFY', updated_at_utc=now())
    protect_sources(stage, absorbed)
    publish(stage, target, absorbed)
    export_controlled(target)
    return read(target / 'consolidation.json')


def export_controlled(target):
    from esrb.measurement.reporting import publish as export
    export(target, target / 'report')


@contextmanager
def lock(path):
    with Path(path).open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def run(repo, job, wait=False, interval=30, natural_only=False):
    job.mkdir(parents=True, exist_ok=True)
    status_path = job / 'status.json'
    state = read(status_path) if status_path.exists() else {'started_at_utc': now()}
    with lock(job / 'lock'):
        try:
            if not state.get('natural_complete'):
                state.update(phase='NATURAL_BUILD', updated_at_utc=now())
                save(status_path, state)
                summary = natural(repo, state)
                state.update(natural_complete=True, natural_episode_status=summary['episode_status'])
                save(status_path, state)
            if natural_only:
                return state
            while not state.get('controlled_complete'):
                base = (repo / '.esr-local/measurement-validation').resolve()
                target = base / 'controlled-20261120-r5'
                if (target / 'consolidation.json').exists() and (target / 'cleanup.json').exists():
                    export_controlled(target)
                    state.update(controlled_complete=True, phase='COMPLETE', completed_at_utc=now(),
                                 controlled_summary=read(target / 'consolidation.json'))
                    save(status_path, state)
                    break
                pending = controlled_ready(base)
                if pending:
                    state.update(phase='WAITING_FOR_RULE_REAUDIT', pending=pending, updated_at_utc=now())
                    save(status_path, state)
                    if not wait:
                        return state
                    time.sleep(interval)
                    continue
                state.update(phase='CONTROLLED_BUILD', updated_at_utc=now())
                save(status_path, state)
                state['controlled_summary'] = controlled(repo, state)
                state.update(controlled_complete=True, phase='COMPLETE', completed_at_utc=now())
                save(status_path, state)
            return state
        except BaseException as error:
            state.update(phase='FAILED', error=f'{type(error).__name__}: {error}', updated_at_utc=now())
            save(status_path, state)
            raise


def main(args):
    print(json.dumps(run(project_root(), args.job, args.wait, args.interval, args.natural_only), ensure_ascii=False, indent=2))
    return 0
