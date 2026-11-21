"""Lightweight CLI composition and frozen-trajectory render dispatch."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import pytest

from esrb.cli.main import build_parser


def test_production_acceptance_defaults_are_explicit():
    args=build_parser().parse_args(['tools','diagnose','production-workcells','--out-dir','NEW'])
    assert args.seeds==list(range(10)) and args.workers==4
    assert args.scenarios==['long_horizon_assembly','moving_transport','moving_assembly']
    assert not args.render


def test_all_three_production_scenes_share_the_render_cli():
    args=build_parser().parse_args(['tools','render','physical-extensions','--scenarios','long_horizon_assembly',
                                  '--acceptance','NEW/verification.json','--out-dir','assets'])
    assert args.scenarios==['long_horizon_assembly']


@pytest.mark.parametrize('scope',['controlled_engineering_acceptance','measured_successful_engineering_trajectories'])
def test_passed_acceptance_uses_saved_records_without_running_the_agent(tmp_path,monkeypatch,scope):
    monkeypatch.setenv('MUJOCO_GL','osmesa')
    pytest.importorskip('mujoco')
    from esrb.tools.render import focused_workcells,physical_extensions
    summary={'scope':scope,'status':'PASS','scenarios':[],'episodes':[]}
    path=tmp_path/'verification.json'; path.write_text(json.dumps(summary))
    render=Mock(); monkeypatch.setattr(focused_workcells,'render_acceptance',render)
    args=SimpleNamespace(width=960,height=720,duration=5.,acceptance=path,scenarios=None,out_dir=tmp_path/'assets')
    physical_extensions.main(args)
    render.assert_called_once_with(summary,args.out_dir,960,720,5.)
