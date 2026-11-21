"""Controlled safety tests must not become Agent-regression discoveries."""
import importlib.util
import json
from pathlib import Path

import pytest

REPORT = __import__('esrb.tools.reports.gate', fromlist=['export'])


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


def test_controlled_regressions_are_reported_separately_from_agent_candidates(tmp_path):
    write(tmp_path/'protocol.json',{'safety_gate':{'enabled':True}})
    for version,kind in [(1,'controlled_acceptance'),(2,'candidate')]:
        folder=tmp_path/'round_01'/f'version_{version:02d}'
        write(folder/'proposal.json',{'kind':kind})
        write(folder/'execution/result.json',{'round':1,'native_task_success':True,'program_ok':True,'safety':{'online_stopped':True,'physics_steps':2}})
        write(folder/'gate_decision.json',{'version':version,'seed':7,'safety_status':'VIOLATION','decision':'REJECT',
              'reason':'observed_safety_violation','parent_before':{'version':1},'parent_after':{'version':1},
              'retained_parent':True,'observed_safety_regression':True})
    result=REPORT.export(tmp_path)
    assert result['controlled_safety_regressions']==1
    assert result['agent_observed_safety_regressions']==1
    assert result['agent_candidate_cases']==1 and result['recorded_decisions']==2


def test_export_cannot_relabel_historical_ungated_data(tmp_path):
    write(tmp_path/'protocol.json',{})
    with pytest.raises(ValueError,match='cannot be relabeled'):REPORT.export(tmp_path)
