"""Only same-trial task/API diagnostics reach an initially empty worker."""
import importlib,json
from pathlib import Path

ROOT=Path(__file__).parents[2]
def module(name):
 group='runners' if name=='run_fresh_agent_trials' else 'feedback'
 return importlib.import_module(f'scripts.{group}.{name}')


def test_standard_errors_exclude_host_diagnostics(tmp_path):
 exporter=module('watch_fresh_agent_api_errors')
 source=tmp_path/'source';source.mkdir()
 for error in [
  {'code':'execution_failed','details':{'exception_type':'ValueError'},'message':'ESRB safety violation'},
  {'code':'adapter_failed','details':{},'message':'forbidden_contact gripper/bin'},
  {'code':'execution_failed','details':{'exception_type':'NameError'},'message':'safety interlock'},
 ]:
  (source/'program-result.json').write_text(json.dumps({'error':error}))
  assert exporter.extract(tmp_path) is None


def test_standard_python_error_is_preserved(tmp_path):
 exporter=module('watch_fresh_agent_api_errors');source=tmp_path/'source';source.mkdir()
 (source/'program-result.json').write_text(json.dumps({'error':{'code':'execution_failed','details':{'exception_type':'NameError'},'message':"program execution failed: NameError: name 'isinstance' is not defined"}}))
 assert exporter.extract(tmp_path)['python_error']['type']=='NameError'
 (source/'program-result.json').write_text(json.dumps({'error':{'code':'invalid_request','details':{},'message':'program rejected: private and dunder names are not allowed'}}))
 assert exporter.extract(tmp_path)=={'program_api_error':'private and dunder names are not allowed'}


def test_base_feedback_does_not_copy_native_or_gate_state(tmp_path):
 bridge=module('run_fresh_agent_trials')
 execution=tmp_path/'execution';execution.mkdir();native=tmp_path/'native';(native/'source').mkdir(parents=True);(native/'trace/calls').mkdir(parents=True)
 (execution/'result.json').write_text(json.dumps({'native_run':str(native),'native_task_success':False,'program_ok':True,'control_steps':4,'safety':{'status':'VIOLATION','force':99},'parent_before':{'version':1}}))
 (native/'source/program-result.json').write_text(json.dumps({'calls':[{'tool':'move_to_pose','ok':False,'error_type':'ValueError','error_message':'ESRB safety violation','outputs_summary':{'ok':False,'steps_executed':4,'force':99}}]}))
 data=bridge.feedback(tmp_path)
 assert data['tool_results']==[{'tool':'move_to_pose','returned':False,'ok':False,'steps_executed':4}]
 assert all(word not in json.dumps(data).lower() for word in ['esrb','safety','violation','parent','force'])


def test_public_error_supplement_blocks_host_stop_reasons(tmp_path,monkeypatch):
 monkeypatch.syspath_prepend(str(ROOT/'scripts'))
 exporter=module('watch_fresh_agent_api_errors_v2')
 (tmp_path/'source').mkdir()
 (tmp_path/'source/program-result.json').write_text('{}')
 calls=tmp_path/'trace/calls';calls.mkdir(parents=True)
 for index,error in enumerate([
  {'code':'timeout','message':'trajectory final pose outside tolerance'},
  {'code':'execution_failed','message':'ESRB safety interlock forbidden_contact force=328'},
  {'code':'provider_failed','message':'gate rejected candidate'},
  {'code':'host_internal','message':'physical force=328'},
 ]):
  folder=calls/str(index);folder.mkdir();(folder/'output.json').write_text(json.dumps({'error':error}))
 assert exporter.task_errors(tmp_path)=={'program_api_error':None,'public_tool_errors':[{'call':'0','code':'timeout','message':'trajectory final pose outside tolerance'}]}
