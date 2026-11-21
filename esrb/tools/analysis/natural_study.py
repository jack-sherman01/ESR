"""Read-only, lineage-aware study of a completed natural campaign and audit.

No detector is rerun here. PASS always refers to the frozen, assessed rules.
The output is a derived paper artifact, never another audit revision.
"""
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import platform
import subprocess
import sys

from esrb.runtime.io import digest, fingerprint, read, save

TASKS = ['stack', 'pickplacecan', 'two-arm-handover', 'nut-assembly', 'wipe']
LABELS = dict(zip(TASKS, ['Stack', 'PickPlaceCan', 'TwoArmHandover', 'NutAssemblySquare', 'Wipe']))
STATUSES = {'PASS', 'VIOLATION', 'UNKNOWN', 'NOT_EVALUATED'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_table(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or (list(rows[0]) if rows else [])
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False, sort_keys=True) if isinstance(v, (list, dict))
                             else v for k, v in row.items()})


def mean(values):
    return sum(values) / len(values) if values else None


def fraction(numerator, denominator):
    return {'numerator': numerator, 'denominator': denominator,
            'value': numerator / denominator if denominator else None}


def predicate_changes(parent, child):
    """Unknown and unexecuted episodes cannot establish a predicate transition."""
    groups = {'new': [], 'resolved': [], 'persistent': [], 'unknown': []}
    for predicate in sorted(set(parent['predicates']) | set(child['predicates'])):
        old, new = parent['predicates'].get(predicate), child['predicates'].get(predicate)
        if parent['audit_status'] == 'NOT_EVALUATED' or child['audit_status'] == 'NOT_EVALUATED':
            groups['unknown'].append(predicate)
        elif old == 'PASS' and new == 'VIOLATION':
            groups['new'].append(predicate)
        elif old == 'VIOLATION' and new == 'PASS':
            groups['resolved'].append(predicate)
        elif old == new == 'VIOLATION':
            groups['persistent'].append(predicate)
        elif old not in {'PASS', 'VIOLATION'} or new not in {'PASS', 'VIOLATION'}:
            groups['unknown'].append(predicate)
    return groups


def matching_events(result, predicates):
    """Select witnesses, not counts of aliases for the same physical contact."""
    found = {}
    for event in result['events']:
        if event['kind'] == 'unconfirmed_drop':
            continue
        groups = '|'.join(sorted(event.get('groups', event.get('contact', {}).get('groups', []))))
        aliases = {event['kind'], 'contact:' + groups, event['kind'] + ':groups=' + groups}
        if aliases.intersection(predicates):
            contact = event.get('contact', event)
            key = (contact.get('chunk'), contact.get('contact_row'), event.get('physics_index'),
                   None if 'contact_row' in contact else event['kind'])
            found.setdefault(key, event)
    return list(found.values())


def accepted_intervals(events, lookup, task, chain, seeds):
    accepted = [event for event in events if event['accepted']]
    rows = []
    for seed in seeds:
        values = [lookup[(task, chain, event['version'], seed)] for event in accepted]
        predicates = sorted(set().union(*(r['predicates'] for r in values))) if values else []
        for predicate in predicates:
            codes = [r['predicates'].get(predicate, 'UNKNOWN') if r['audit_status'] not in {'UNKNOWN', 'NOT_EVALUATED'}
                     else 'UNKNOWN' for r in values]
            pos, previous_end = 0, None
            while pos < len(codes):
                if codes[pos] != 'VIOLATION':
                    pos += 1
                    continue
                start = pos
                while pos < len(codes) and codes[pos] == 'VIOLATION':
                    pos += 1
                gap = codes[previous_end:start] if previous_end is not None else []
                rows.append({'task': task, 'chain': chain, 'seed': seed, 'predicate': predicate,
                    'start_version': accepted[start]['version'], 'last_violating_version': accepted[pos-1]['version'],
                    'accepted_parent_states': pos-start,
                    'resolution_version': accepted[pos]['version'] if pos < len(codes) and codes[pos] == 'PASS' else None,
                    'recurrence_after_known_pass': bool(gap and all(c == 'PASS' for c in gap)),
                    'present_in_first_parent': start == 0,
                    'end_status': codes[pos] if pos < len(codes) else 'RIGHT_CENSORED_AT_FINAL_PARENT'})
                previous_end = pos
    return rows


class Inputs:
    """Hash every consumed input, then check it again after producing the report."""
    def __init__(self):
        self.files = {}

    def bind(self, path, expected=None):
        path = Path(path).absolute()
        if str(path) not in self.files:
            self.files[str(path)] = digest(path)
        sha = self.files[str(path)]
        require(expected is None or sha == expected, 'Input SHA256 mismatch: ' + str(path))
        return path

    def json(self, path, expected=None):
        return read(self.bind(path, expected))

    def text(self, path, expected=None):
        return self.bind(path, expected).read_text()

    def verify(self):
        for path, sha in self.files.items():
            require(digest(path) == sha, 'Input changed during analysis: ' + path)


def load_study(campaign, audit, inputs):
    import numpy as np
    manifest = inputs.json(campaign / 'manifest.json')
    completion = inputs.json(campaign / 'completion.json')
    protocol = inputs.json(audit / 'protocol.json')
    index = inputs.json(audit / 'inputs.json', protocol['inputs_sha256'])
    inputs.bind(audit / 'rules.json', protocol['rules_sha256'])
    require(Path(index['campaign']).resolve() == campaign.resolve(), 'Campaign/audit mismatch')
    native_chains = {(c['task'], c['chain']): c for c in manifest['chains']}
    require(len(native_chains) == len(manifest['chains']) == len(index['chains']), 'Duplicate/missing chain')
    lookup, native, episodes, chains = {}, {}, [], []
    for chain in index['chains']:
        key = (chain['task'], chain['chain'])
        require(key in native_chains, 'Audit chain absent from campaign')
        root = Path(native_chains[key]['root'])
        require(root.resolve() == Path(chain['root']).resolve(), 'Chain root mismatch')
        state = inputs.json(root / 'state.json', chain['state_sha256'])
        cp = inputs.json(root / 'protocol.json', chain['protocol_sha256'])
        if 'api_sha256' in cp:
            inputs.bind(root / 'API.txt', cp['api_sha256'])
        if 'selection' in cp:
            require(cp['selection'] == 'task-only-all-development-cases' and cp['capture'] == 'raw-only', 'Unexpected intervention protocol')
            require(cp['physics_timestep_s'] == 0.002 and cp['control_frequency_hz'] == 20, 'Unexpected physics/control configuration')
        inputs.bind(root / 'artifacts.json', chain['artifact_manifest_sha256'])
        require(state['events'] == chain['events'], 'Frozen/native lineage mismatch')
        seeds = cp['seeds']
        require(len(seeds) == len(set(seeds)) and seeds == [1, 2, 3, 4, 5], 'Unexpected development seeds')
        require(len(state['events']) == cp['candidate_budget'], 'Incomplete candidate slots')
        active, previous = None, None
        last_program = None
        for version, event in enumerate(state['events'], 1):
            require(event['version'] == version and event['parent_before'] == active, 'Incorrect actual parent')
            require(event['generation_base'] == (active or last_program), 'Generation did not start from the recorded eligible parent/recent acquisition program')
            require(event['previous_event_sha256'] == previous and event['event_sha256'] ==
                    fingerprint({k: v for k, v in event.items() if k != 'event_sha256'}), 'Lineage hash mismatch')
            require(event['program'] is not None and sorted(c['seed'] for c in event['cases']) == seeds,
                    'Missing program or development cases')
            inputs.bind(root / event['program'], event['program_sha256'])
            inputs.bind(root / f'candidates/v{version:02d}/proposal.json')
            require(event['accepted'] == all(c['task_qualified'] for c in event['cases']), 'Acceptance is not 5/5')
            expected = ({'version': version, 'program': event['program'], 'program_sha256': event['program_sha256']}
                        if event['accepted'] else active)
            require(event['parent_after'] == expected, 'Rejected candidate changed the parent')
            active, previous = expected, event['event_sha256']
            last_program = {'version': version, 'program': event['program'], 'program_sha256': event['program_sha256']}
        require(active == state['active'] and previous == state['previous_event_sha256'], 'Final state mismatch')
        chains.append(dict(chain, root=str(root), seeds=seeds, protocol=cp))
    expected_cases = {(c['task'], c['chain'], e['version'], s) for c in chains for e in c['events'] for s in c['seeds']}
    references = {}
    for item in index['episodes']:
        key = (item['task'], item['chain'], item['version'], item['seed'])
        require(key not in lookup, 'Duplicate episode identity')
        path = audit / 'episodes' / item['task'] / f"chain_{item['chain']:02d}" / f"v{item['version']:02d}_seed_{item['seed']}.json"
        result = inputs.json(path)
        execution = Path(item['execution'])
        raw = inputs.json(execution / 'result.json', item['result_sha256'])
        identity = inputs.json(execution / 'initial_identity.json')
        inputs.bind(execution / 'model.xml', item['model_sha256'])
        state_path = inputs.bind(execution / 'initial_state.npy')
        actual_id = fingerprint({'model': item['model_sha256'], 'state': np.load(state_path, allow_pickle=False).tolist()})
        require(actual_id == item['initial_state_id'], 'Raw initial state does not reproduce initial_state_id')
        require(result['audit_status'] in STATUSES and result['id'] == item['id'], 'Invalid audit label or id')
        for field in ['initial_state_id', 'model_sha256', 'program_sha256', 'result_sha256']:
            require(result[field] == item[field], 'Audit identity mismatch: ' + field)
        require(identity['initial_state_id'] == item['initial_state_id'] and identity['seed'] == item['seed']
                and identity['model_sha256'] == item['model_sha256'], 'Actual initial identity mismatch')
        require(raw['initial_state_id'] == item['initial_state_id'] and raw['program_sha256'] == item['program_sha256'],
                'Native execution identity mismatch')
        qualified = raw['program_ok'] and raw['terminal_native_success']
        require(qualified == raw['task_qualified'] == item['task_qualified'] == result['task_qualified'],
                'Task qualification mismatch')
        refkey = (item['task'], item['seed'])
        refvalue = (item['initial_state_id'], item['model_sha256'])
        require(refkey not in references or references[refkey] == refvalue, 'Cross-version/chain initial state mismatch')
        references[refkey] = refvalue
        require(not result['coverage_complete'] or raw['capture']['coverage_complete'], 'Audit claims coverage absent in native capture')
        row = {k: result.get(k) for k in ['id', 'task', 'chain', 'version', 'seed', 'parent_version',
               'task_accepted', 'task_qualified', 'program_ok', 'native_task_success', 'audit_status',
               'coverage_complete', 'control_steps', 'execution_evidence_status', 'execution_reason']}
        row.update(terminal_native_success=raw['terminal_native_success'],
                   native_success_at_any_sample=raw.get('native_success_at_any_sample'),
                   termination_reason=raw.get('termination_reason'), wall_seconds=raw.get('wall_seconds'),
                   native_capture_complete=raw['capture']['coverage_complete'],
                   physics_steps=result.get('metrics', {}).get('physics_steps', 0),
                   initial_state_id=item['initial_state_id'], model_sha256=item['model_sha256'],
                   violation_kinds=sorted({e['kind'] for e in result['events'] if e['kind'] != 'unconfirmed_drop'}),
                   execution=str(execution), audit_file=str(path))
        lookup[key], native[key] = result, raw
        episodes.append(row)
    require(set(lookup) == expected_cases, 'Episode set does not match chain events')
    require(len({r['id'] for r in episodes}) == len(episodes), 'Duplicate episode ids')
    for chain in chains:
        for event in chain['events']:
            for case in event['cases']:
                r = lookup[(chain['task'], chain['chain'], event['version'], case['seed'])]
                require(r['task_accepted'] == event['accepted'] and r['parent_version'] ==
                        (event['parent_before'] or {}).get('version') and r['program_sha256'] == event['program_sha256'],
                        'Audit/native parent or program mismatch')
                require(r['task_qualified'] == case['task_qualified'], 'Case qualification mismatch')
    for task in {c['task'] for c in chains}:
        require(len({v[0] for (t, _), v in references.items() if t == task}) == 5, 'Seeds do not vary initial states')
    return chains, lookup, native, episodes, protocol, completion


def analyze(chains, lookup, native):
    candidates, updates, pairs, chain_rows, seed_rows, intervals, failure_rows = [], [], [], [], [], [], []
    for chain in chains:
        task, number, seeds = chain['task'], chain['chain'], chain['seeds']
        local_candidates, local_updates = [], []
        accepted = [e for e in chain['events'] if e['accepted']]
        for event in chain['events']:
            version = event['version']
            child = [lookup[(task, number, version, seed)] for seed in seeds]
            counts = Counter(r['audit_status'] for r in child)
            candidate = {'task': task, 'chain': number, 'version': version,
                'generation_base': (event['generation_base'] or {}).get('version'),
                'parent_version': (event['parent_before'] or {}).get('version'), 'task_accepted': event['accepted'],
                'task_success_cases': sum(r['task_qualified'] for r in child), 'violation_cases': counts['VIOLATION'],
                'assessed_pass_cases': counts['PASS'], 'unknown_cases': counts['UNKNOWN'],
                'not_evaluated_cases': counts['NOT_EVALUATED'],
                'mean_control_steps': mean([r['control_steps'] for r in child]),
                'context_revision_effective_from': (chain.get('context_revision') or {}).get('effective_from_version')}
            candidates.append(candidate); local_candidates.append(candidate)
            if not event['parent_before']:
                continue
            parent_version = event['parent_before']['version']
            parent = [lookup[(task, number, parent_version, s)] for s in seeds]
            new, resolved, persistent, unknown = set(), set(), set(), set()
            strict = all(r['audit_status'] == 'PASS' for r in parent)
            strict_known = strict and (counts['VIOLATION'] > 0 or all(r['audit_status'] == 'PASS' for r in child))
            common = []
            for old, newr in zip(parent, child):
                require(old['initial_state_id'] == newr['initial_state_id'] and old['model_sha256'] == newr['model_sha256'],
                        'Unpaired actual identity')
                changes = predicate_changes(old, newr)
                new.update(changes['new']); resolved.update(changes['resolved'])
                persistent.update(changes['persistent']); unknown.update(changes['unknown'])
                common_success = old['task_qualified'] and newr['task_qualified']
                if common_success:
                    common.append((old['control_steps'], newr['control_steps']))
                pairs.append({'task': task, 'chain': number, 'version': version, 'parent_version': parent_version,
                    'seed': newr['seed'], 'task_accepted': event['accepted'], 'paired_identity': True,
                    'parent_audit': old['audit_status'], 'child_audit': newr['audit_status'],
                    'new_predicates': changes['new'], 'resolved_predicates': changes['resolved'],
                    'persistent_predicates': changes['persistent'], 'unknown_predicates': changes['unknown'],
                    'common_task_success': common_success,
                    'parent_steps': old['control_steps'], 'child_steps': newr['control_steps'],
                    'delta_steps_common_success': newr['control_steps']-old['control_steps'] if common_success else None})
            parent_mean = mean([p for p, _ in common]); child_mean = mean([c for _, c in common])
            update = {'task': task, 'chain': number, 'version': version, 'parent_version': parent_version,
                'task_accepted': event['accepted'], 'new_violation': bool(new), 'new_predicates': sorted(new),
                'resolved_predicates': sorted(resolved), 'persistent_predicates': sorted(persistent),
                'unknown_predicates': sorted(unknown), 'all_five_parent_pass': strict,
                'strict_panel_outcome_known': strict_known,
                'strict_panel_new_violation': strict and counts['VIOLATION'] > 0,
                'parent_violation_cases': sum(r['audit_status'] == 'VIOLATION' for r in parent),
                'child_violation_cases': counts['VIOLATION'], 'common_success_cases': len(common),
                'parent_mean_steps_common': parent_mean, 'child_mean_steps_common': child_mean,
                'relative_steps_change_common': child_mean/parent_mean-1 if parent_mean else None}
            updates.append(update); local_updates.append(update)
        first = accepted[0]['version'] if accepted else None
        final = accepted[-1]['version'] if accepted else None
        def panel(version):
            return [lookup[(task, number, version, s)] for s in seeds] if version else []
        fp, lp = panel(first), panel(final)
        fsteps, lsteps = mean([r['control_steps'] for r in fp]), mean([r['control_steps'] for r in lp])
        bad_history = any(c['task_accepted'] and c['violation_cases'] for c in local_candidates)
        chain_rows.append({'task': task, 'chain': number, 'candidate_slots': len(chain['events']),
            'task_accepted': len(accepted), 'first_qualified_version': first, 'final_parent_version': final,
            'task_qualified_episodes': sum(c['task_success_cases'] for c in local_candidates),
            'first_parent_violation_cases': sum(r['audit_status'] == 'VIOLATION' for r in fp) if fp else None,
            'final_parent_violation_cases': sum(r['audit_status'] == 'VIOLATION' for r in lp) if lp else None,
            'final_parent_all_assessed_pass': all(r['audit_status'] == 'PASS' for r in lp) if lp else None,
            'accepted_unsafe_candidates': sum(c['task_accepted'] and c['violation_cases'] > 0 for c in local_candidates),
            'first_parent_mean_steps': fsteps, 'final_parent_mean_steps': lsteps,
            'relative_endpoint_steps_change': lsteps/fsteps-1 if fsteps else None,
            'proposed_new_violation_updates': sum(u['new_violation'] for u in local_updates),
            'accepted_new_violation_updates': sum(u['new_violation'] and u['task_accepted'] for u in local_updates),
            'first_proposed_regression_version': next((u['version'] for u in local_updates if u['new_violation']), None),
            'first_accepted_regression_version': next((u['version'] for u in local_updates if u['new_violation'] and u['task_accepted']), None),
            'final_clean_masks_earlier_accepted_violation': bool(lp and all(r['audit_status'] == 'PASS' for r in lp) and bad_history),
            'context_revision_effective_from': (chain.get('context_revision') or {}).get('effective_from_version')})
        intervals.extend(accepted_intervals(chain['events'], lookup, task, number, seeds))
        for seed in seeds:
            values = [lookup[(task, number, e['version'], seed)] for e in chain['events']]
            seed_rows.append({'task': task, 'chain': number, 'seed': seed, 'episodes': len(values),
                'task_qualified': sum(r['task_qualified'] for r in values),
                'violations': sum(r['audit_status'] == 'VIOLATION' for r in values),
                'not_evaluated': sum(r['audit_status'] == 'NOT_EVALUATED' for r in values)})
            reasons = Counter(native[(task, number, e['version'], seed)].get('termination_reason', 'UNRECORDED')
                              for e in chain['events'] if not lookup[(task, number, e['version'], seed)]['task_qualified'])
            for reason, n in sorted(reasons.items()):
                failure_rows.append({'task': task, 'chain': number, 'seed': seed, 'termination_reason': reason, 'episodes': n})
    return dict(candidates=candidates, updates=updates, paired_cases=pairs, chains=chain_rows,
                seed_performance=seed_rows, lineage_intervals=intervals, failure_reasons=failure_rows)


def summary_tables(episodes, tables, lookup):
    scenes, cross, rates, hazard_rows = [], [], [], []
    for task in TASKS + ['ALL']:
        es = [r for r in episodes if task == 'ALL' or r['task'] == task]
        cs = [r for r in tables['candidates'] if task == 'ALL' or r['task'] == task]
        us = [r for r in tables['updates'] if task == 'ALL' or r['task'] == task]
        ch = [r for r in tables['chains'] if task == 'ALL' or r['task'] == task]
        counts = Counter(r['audit_status'] for r in es)
        scenes.append({'task': task, 'chains': len(ch), 'episodes': len(es), 'candidates': len(cs),
            'task_qualified_episodes': sum(r['task_qualified'] for r in es),
            'native_success_episodes': sum(r['native_task_success'] for r in es),
            'accepted_candidates': sum(r['task_accepted'] for r in cs),
            'chains_with_task_parent': sum(r['first_qualified_version'] is not None for r in ch),
            'pass_episodes': counts['PASS'], 'violation_episodes': counts['VIOLATION'],
            'unknown_episodes': counts['UNKNOWN'], 'not_evaluated_episodes': counts['NOT_EVALUATED'],
            'unsafe_candidates': sum(r['violation_cases'] > 0 for r in cs),
            'accepted_unsafe_candidates': sum(r['task_accepted'] and r['violation_cases'] > 0 for r in cs),
            'parent_updates': len(us), 'new_violation_updates': sum(r['new_violation'] for r in us),
            'accepted_new_violation_updates': sum(r['new_violation'] and r['task_accepted'] for r in us),
            'strict_clean_parent_updates': sum(r['all_five_parent_pass'] for r in us),
            'strict_known_parent_updates': sum(r['strict_panel_outcome_known'] for r in us),
            'strict_new_violation_updates': sum(r['strict_panel_new_violation'] for r in us),
            'accepted_strict_new_violation_updates': sum(r['strict_panel_new_violation'] and r['task_accepted'] for r in us)})
        metrics = {
            'task_qualification': (sum(r['task_qualified'] for r in es), len(es)),
            'episode_violation_assessable': (counts['VIOLATION'], counts['PASS']+counts['VIOLATION']),
            'candidate_with_observed_violation': (sum(c['violation_cases'] > 0 for c in cs), len(cs)),
            'accepted_candidate_with_observed_violation': (sum(c['task_accepted'] and c['violation_cases'] > 0 for c in cs), sum(c['task_accepted'] for c in cs)),
            'new_predicate_violation_observed_all_updates': (sum(u['new_violation'] for u in us), len(us)),
            'strict_panel_new_violation': (sum(u['strict_panel_new_violation'] for u in us), sum(u['strict_panel_outcome_known'] for u in us)),
            'accepted_update_with_new_predicate_violation': (sum(u['new_violation'] and u['task_accepted'] for u in us), sum(u['task_accepted'] for u in us)),
        }
        for name, (n, d) in metrics.items():
            rates.append(dict(task=task, metric=name, **fraction(n, d)))
        for qualified in [False, True]:
            for status in sorted(STATUSES):
                cross.append({'task': task, 'task_qualified': qualified, 'audit_status': status,
                              'episodes': sum(r['task_qualified'] == qualified and r['audit_status'] == status for r in es)})
        event_groups = defaultdict(set)
        for r in es:
            result = lookup[(r['task'], r['chain'], r['version'], r['seed'])]
            for e in result['events']:
                if e['kind'] != 'unconfirmed_drop':
                    event_groups[e['kind']].add((r['task'], r['chain'], r['version'], r['seed']))
        for kind, ids in sorted(event_groups.items()):
            hazard_rows.append({'task': task, 'event_kind': kind, 'episodes': len(ids),
                'candidates': len({i[:3] for i in ids}), 'chains': len({i[:2] for i in ids}),
                'counting_unit': 'unique units containing this event kind; rows overlap, not accident counts'})
    tables.update(scenes=scenes, task_audit_cross=cross, rates=rates, hazard_types=hazard_rows)


def check_cached_tables(audit, tables, inputs):
    differences = []
    # Derived caches are checked, never trusted over current episode-level labels.
    for name in ['candidates', 'updates']:
        p = audit / (name + '.csv')
        if not p.exists():
            continue
        inputs.bind(p)
        with p.open() as stream:
            old = {(r['task'], int(r['chain']), int(r['version'])): r for r in csv.DictReader(stream)}
        for row in tables[name]:
            prior = old.get((row['task'], row['chain'], row['version']), {})
            for field in ['unknown_cases', 'task_success_cases', 'violation_cases', 'assessed_pass_cases', 'new_violation']:
                if field in row and field in prior and str(row[field]) != prior[field]:
                    differences.append({'table': name, 'task': row['task'], 'chain': row['chain'], 'version': row['version'],
                                        'field': field, 'cached': prior[field], 'rebuilt': row[field]})
    return differences


def main(args):
    campaign, audit, out = args.campaign.absolute(), args.audit.absolute(), args.out.absolute()
    for protected in [campaign, audit]:
        require(not out.resolve().is_relative_to(protected.resolve()) and not protected.resolve().is_relative_to(out.resolve()),
                'Output must be separate from campaign/audit inputs')
    require(not out.exists() or not any(out.iterdir()), 'Output directory must be empty; choose a new output or remove only previous derived output')
    plot_python = str(args.plot_python or sys.executable)
    probe = subprocess.run([plot_python, '-c', 'import matplotlib; print(matplotlib.__version__)'], capture_output=True, text=True)
    require(probe.returncode == 0, 'Plot interpreter needs matplotlib; use --plot-python /path/to/python')
    inputs = Inputs()
    chains, lookup, native, episodes, protocol, completion = load_study(campaign, audit, inputs)
    for chain in chains:
        protected = Path(chain['root']).resolve()
        require(not out.resolve().is_relative_to(protected) and not protected.is_relative_to(out.resolve()),
                'Output overlaps a native chain')
    tables = analyze(chains, lookup, native)
    summary_tables(episodes, tables, lookup)
    tables['episodes'] = episodes
    coverage = Counter()
    for r in lookup.values():
        for predicate, status in r['predicates'].items():
            effective = 'NOT_EVALUATED' if r['audit_status'] == 'NOT_EVALUATED' else status
            coverage[(r['task'], 'assessed_predicate', predicate, effective)] += 1
        for risk, status in r.get('unassessed', {}).items():
            coverage[(r['task'], 'unassessed_risk', risk, status)] += 1
    tables['rule_coverage'] = [dict(zip(['task', 'category', 'rule', 'status'], key), episodes=n)
                               for key, n in sorted(coverage.items())]
    tables['data_quality'] = [
        {'check': 'unique_episode_ids', 'count': len(episodes), 'denominator': len(episodes)},
        {'check': 'coverage_complete', 'count': sum(r['coverage_complete'] for r in episodes), 'denominator': len(episodes)},
        {'check': 'dynamic_physics_trace', 'count': sum(r['physics_steps'] > 0 for r in episodes), 'denominator': len(episodes)},
        {'check': 'native_success_but_not_qualified', 'count': sum(r['native_task_success'] and not r['task_qualified'] for r in episodes), 'denominator': len(episodes)},
        {'check': 'paired_initial_model_identity', 'count': len(tables['paired_cases']), 'denominator': len(tables['paired_cases'])}]
    differences = check_cached_tables(audit, tables, inputs)
    tables['cached_table_differences'] = differences
    out.mkdir(parents=True, exist_ok=True)
    from .natural_study_evidence import build_cases, continuous_tables
    cases, checks = build_cases(chains, lookup, native, tables, out, inputs)
    tables.update(continuous_tables(audit, lookup, inputs))
    for name, rows in tables.items():
        write_table(out / 'tables' / (name + '.csv'), rows)
    save(out / 'cases.json', cases)
    plot_names = ['chains', 'candidates', 'updates', 'scenes', 'task_audit_cross', 'rates']
    save(out / 'analysis.json', {'schema': 'esrb.natural_study/1',
                                'tables': {name: tables[name] for name in plot_names},
                                'table_files': {name: 'tables/' + name + '.csv' for name in tables},
                                'cases_file': 'cases.json',
                                'cases': [c for c in cases if c['selection'] == 'new_violation' and c['task_accepted'] and c['all_five_parent_pass']],
                                'scope': 'post-hoc development seeds 1-5; frozen assessed rules only'})
    plot_script = Path(__file__).with_name('natural_study_plots.py')
    subprocess.run([plot_python, str(plot_script), str(out)], check=True)
    from .natural_study_report import report, dictionary
    (out / 'REPORT.md').write_text(report(tables, cases, checks, campaign, audit), encoding='utf-8')
    (out / 'DATA_DICTIONARY.md').write_text(dictionary(), encoding='utf-8')
    for source in [Path(__file__), plot_script, Path(__file__).with_name('natural_study_evidence.py'),
                   Path(__file__).with_name('natural_study_report.py'), Path(__file__).with_name('natural_study_notes.py')]:
        inputs.bind(source)
    inputs.verify()
    manifest = {'schema': 'esrb.natural_study_inputs/1', 'campaign': str(campaign), 'audit': str(audit),
        'rules_sha256': protocol['rules_sha256'], 'files': dict(sorted(inputs.files.items())),
        'python': platform.python_version(), 'plot_python': plot_python, 'matplotlib': probe.stdout.strip(),
        'new_physics_episodes': 0, 'new_model_calls': 0}
    save(out / 'input-manifest.json', manifest)
    save(out / 'validation.json', {'status': 'PASS', 'episodes': len(episodes), 'candidates': len(tables['candidates']),
        'chains': len(chains), 'parent_updates': len(tables['updates']), 'paired_cases': len(tables['paired_cases']),
        'input_files_unchanged': len(inputs.files), 'raw_witness_checks': checks,
        'cached_table_differences': len(differences), 'new_physics_episodes': 0, 'new_model_calls': 0})
    save(out / 'output-manifest.json', {str(p.relative_to(out)): digest(p) for p in sorted(out.rglob('*')) if p.is_file()})
    print(json.dumps({'status': 'PASS', 'report': str(out / 'REPORT.md'), 'episodes': len(episodes),
                      'regression_cases': sum(c['selection'] == 'new_violation' for c in cases)}, ensure_ascii=False))
