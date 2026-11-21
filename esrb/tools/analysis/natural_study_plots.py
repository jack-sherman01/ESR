"""Standalone plotting worker: only JSON + matplotlib, compatible with Python 3.8."""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

LABELS = {'stack': 'Stack', 'pickplacecan': 'PickPlaceCan', 'two-arm-handover': 'TwoArmHandover',
          'nut-assembly': 'NutAssemblySquare', 'wipe': 'Wipe'}
COLORS = {'stack': '#2878b5', 'pickplacecan': '#e48727', 'two-arm-handover': '#54a24b',
          'nut-assembly': '#b279a2', 'wipe': '#79706e'}


def save(fig, root, name):
    fig.savefig(str(root / (name + '.png')), dpi=180, bbox_inches='tight', metadata={'Software': 'ESRB natural-study'})
    fig.savefig(str(root / (name + '.pdf')), bbox_inches='tight', metadata={'CreationDate': None, 'ModDate': None})
    plt.close(fig)


def draw(root):
    data = json.loads((root / 'analysis.json').read_text())
    t = data['tables']; chains = t['chains']; candidates = t['candidates']; updates = t['updates']
    figs = root / 'figures'; figs.mkdir(exist_ok=True)
    labels = [LABELS[c['task']] + ' / ' + str(c['chain']) for c in chains]
    width = max(c['version'] for c in candidates)
    taskmat = np.full((len(chains), width), np.nan); riskmat = taskmat.copy(); parentmat = taskmat.copy()
    unknown = []; revision = []
    for i, chain in enumerate(chains):
        rows = [c for c in candidates if (c['task'], c['chain']) == (chain['task'], chain['chain'])]
        active = None
        for row in rows:
            j = row['version'] - 1
            taskmat[i, j], riskmat[i, j] = row['task_success_cases'], row['violation_cases']
            if row['not_evaluated_cases'] or row['unknown_cases']:
                unknown.append((j, i))
            if row['task_accepted']:
                active = row['version']
            parentmat[i, j] = active if active is not None else np.nan
        if chain['context_revision_effective_from']:
            revision.append((chain['context_revision_effective_from']-1, i))
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True, constrained_layout=True)
    for ax, matrix, cmap, title, vmax in zip(axes, [taskmat, riskmat, parentmat], ['Blues', 'YlOrRd', 'viridis'],
        ['Task-qualified seeds / 5', 'Seeds with observed assessed-rule violations / 5', 'Active task parent after candidate (blank: none)'], [5, 5, width]):
        im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=0, vmax=vmax, interpolation='nearest')
        ax.set_yticks(range(len(chains))); ax.set_yticklabels(labels, fontsize=8)
        ax.set_title(title); fig.colorbar(im, ax=ax, pad=0.01)
        for x, y in revision:
            ax.plot(x, y, '|', color='black', markersize=10)
    for x, y in unknown:
        axes[1].plot(x, y, 's', markerfacecolor='none', markeredgecolor='#555555', markersize=5)
    axes[-1].set_xticks(list(range(0, width, 5))); axes[-1].set_xticklabels(list(range(1, width+1, 5)))
    axes[-1].set_xlabel('Candidate version; black tick: context revision; outlined risk cell: UNKNOWN or NOT_EVALUATED present')
    save(fig, figs, 'version_evolution')

    # Counts are shown rather than percentages that could hide unexecuted cases.
    scenes = [s for s in t['scenes'] if s['task'] != 'ALL']
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    bottoms = np.zeros(len(scenes)); xs = np.arange(len(scenes))
    for qualified, status, color, label in [(True, 'PASS', '#54a24b', 'Task success / scoped PASS'),
        (True, 'VIOLATION', '#e45756', 'Task success / violation'), (False, 'PASS', '#9ecae9', 'Task failure / scoped PASS'),
        (False, 'VIOLATION', '#f2a541', 'Task failure / violation')]:
        values = [next(r['episodes'] for r in t['task_audit_cross'] if r['task'] == s['task'] and
                       r['task_qualified'] == qualified and r['audit_status'] == status) for s in scenes]
        ax.bar(xs, values, bottom=bottoms, color=color, label=label); bottoms += values
    missing = [s['not_evaluated_episodes'] + s['unknown_episodes'] for s in scenes]
    ax.bar(xs, missing, bottom=bottoms, color='#888888', label='NOT_EVALUATED / UNKNOWN')
    ax.set_xticks(xs); ax.set_xticklabels([LABELS[s['task']] for s in scenes], fontsize=9)
    ax.set_ylabel('Development episodes'); ax.set_title('Task qualification and retrospective assessed-rule outcomes')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=9)
    save(fig, figs, 'task_audit_cross')

    fig, axes = plt.subplots(5, 3, figsize=(17, 13), sharex=True, constrained_layout=True)
    for ax, chain in zip(axes.flat, chains):
        rows = [c for c in candidates if (c['task'], c['chain']) == (chain['task'], chain['chain'])]
        accepted = [r for r in rows if r['task_accepted']]
        if accepted:
            ax.step([r['version'] for r in accepted], [r['violation_cases'] for r in accepted], where='post', color='#333333', linewidth=1)
        ax.scatter([r['version'] for r in accepted], [r['violation_cases'] for r in accepted],
                   c=[r['violation_cases'] for r in accepted], cmap='YlOrRd', vmin=0, vmax=5, s=17)
        rejected = [r for r in rows if not r['task_accepted']]
        ax.scatter([r['version'] for r in rejected], [r['violation_cases'] for r in rejected], marker='x', color='#999999', s=17)
        news = [u for u in updates if (u['task'], u['chain']) == (chain['task'], chain['chain']) and u['new_violation']]
        ax.scatter([u['version'] for u in news], [u['child_violation_cases'] for u in news],
                   marker='o', facecolors='none', edgecolors='#2166ac', s=70)
        ax.set_title(LABELS[chain['task']] + ' / ' + str(chain['chain']), fontsize=10)
        ax.set_ylim(-0.5, 5.5); ax.set_yticks([0, 1, 3, 5]); ax.grid(alpha=0.15)
    fig.suptitle('Actual accepted lineage: filled = accepted; x = rejected proposal; blue ring = new predicate violation\n'
                 'Lines connect accepted parents only; y = violating development seeds (not severity)', fontsize=11)
    for ax in axes[-1]: ax.set_xlabel('Candidate version')
    save(fig, figs, 'accepted_lineages')

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for task in LABELS:
        points = [u for u in updates if u['task'] == task and u['relative_steps_change_common'] is not None]
        ax.scatter([100*u['relative_steps_change_common'] for u in points],
                   [u['child_violation_cases']-u['parent_violation_cases'] for u in points],
                   s=22, alpha=0.5, color=COLORS[task], label=LABELS[task])
    points = [u for u in updates if u['new_violation'] and u['task_accepted'] and u['relative_steps_change_common'] is not None]
    ax.scatter([100*u['relative_steps_change_common'] for u in points],
               [u['child_violation_cases']-u['parent_violation_cases'] for u in points],
               facecolors='none', edgecolors='black', s=100, label='Accepted new predicate violation')
    ax.axhline(0, color='#999999', linewidth=0.7); ax.axvline(0, color='#999999', linewidth=0.7)
    ax.set_xlabel('Mean control-step change on common successful seeds (%)')
    ax.set_ylabel('Change in number of violating seeds')
    ax.set_title('Paired task efficiency and observed rule outcomes\nZero count change can conceal a new hazard type; no severity or causal claim')
    ax.legend(fontsize=8); save(fig, figs, 'paired_efficiency')

    featured = [c for c in data['cases'] if c['selection'] == 'new_violation' and c['task_accepted'] and c['all_five_parent_pass']]
    if featured:
        fig, axes = plt.subplots(len(featured), 1, figsize=(14, 3.0*len(featured)), squeeze=False, constrained_layout=True)
        for ax, case in zip(axes[:, 0], featured):
            witness = case['witnesses'][0]; seed = witness['seed']
            pair = next(r for r in case['five_seed_results'] if r['seed'] == seed)
            durations = []
            for i, field in enumerate(['parent_api_timeline', 'child_api_timeline']):
                spans = [s for s in pair[field] if s.get('parent_id') is None and s['start_s'] is not None and s['end_s'] is not None and s['end_s'] > s['start_s']]
                for span in spans:
                    color = '#fdbf6f' if 'gripper' in span['name'] else '#a6cee3' if i == 0 else '#b2df8a'
                    ax.broken_barh([(span['start_s'], span['end_s']-span['start_s'])], (i-0.18, 0.36), facecolors=color, edgecolors='white', linewidth=0.7)
                    durations.append(span['end_s'])
            event = witness['event']; contact = event.get('contact', event)
            when = contact.get('time_start_s', event.get('time_s'))
            if when is not None:
                ax.scatter([when], [1], color='#d62728', marker='*', s=140, zorder=4)
                ax.axvline(when, color='#d62728', alpha=0.4, linewidth=0.8)
            label = '{} C{} V{} -> V{} / seed {}: {}\n{}; first witness Fn = {:.3g} N'.format(
                LABELS[case['task']], case['chain'], case['parent_version'], case['version'], seed,
                event['kind'], ' / '.join(contact.get('geom_names', [])), contact.get('normal_force_n', 0))
            ax.set_title(label, fontsize=9); ax.set_yticks([0, 1]); ax.set_yticklabels(['Parent API', 'Child API'])
            ax.set_ylim(-0.6, 1.6); ax.set_xlabel('Simulation time (s); outlined bars = top-level API spans; orange = gripper; star = checked contact')
        save(fig, figs, 'regression_evidence')

    # Endpoint comparisons are restricted to actual first/final task parents, all five common successes.
    learned = [c for c in chains if c['first_qualified_version'] is not None]
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    x = np.arange(len(learned))
    ax.bar(x-0.18, [c['first_parent_mean_steps'] for c in learned], 0.36, label='First qualified parent')
    ax.bar(x+0.18, [c['final_parent_mean_steps'] for c in learned], 0.36, label='Final qualified parent')
    ax.set_xticks(x); ax.set_xticklabels([LABELS[c['task']]+' / '+str(c['chain']) for c in learned], rotation=35, ha='right', fontsize=8)
    ax.set_ylabel('Mean control steps across five successful seeds'); ax.legend()
    ax.set_title('Task-parent endpoint efficiency; no learned parent in four chains')
    save(fig, figs, 'task_parent_efficiency')


if __name__ == '__main__':
    draw(Path(sys.argv[1]))
