"""Same-trial Python and public motion errors, excluding host-only stop reasons."""

# Allow direct execution from this subdirectory as well as package imports.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pathlib import Path
import json,time,os
from scripts.feedback.watch_fresh_agent_api_errors import ROOT,BLOCKED,extract
ALLOWED_CODES={'invalid_request','unsupported','planning_failed','ik_failed','timeout','execution_failed','service_unavailable','not_found','provider_failed'}

def task_errors(native):
 data=extract(native) or {'program_api_error':None}
 errors=[]
 for p in sorted((native/'trace/calls').glob('*/output.json')):
  output=json.loads(p.read_text());error=output.get('error')
  if not isinstance(error,dict):continue
  code=error.get('code');message=error.get('message','')
  if code not in ALLOWED_CODES or not isinstance(message,str) or BLOCKED.search(message):continue
  errors.append({'call':p.parent.name,'code':code,'message':message[:250]})
 if errors:data['public_tool_errors']=errors[-16:]
 return data

if __name__=='__main__':
 (ROOT/'additional-feedback-v2.pid').write_text(str(os.getpid())+'\n')
 while True:
  for scene in ['pickup_can','wipe-formal']:
   for assignment in (ROOT/scene).glob('round_*/agent_assignment.json'):
    capsule=Path(json.loads(assignment.read_text())['capsule'])
    for result_path in assignment.parent.glob('version_*/execution/result.json'):
     version=int(result_path.parent.parent.name[-2:]);out=capsule/f'additional_feedback_v{version:02d}.json'
     if out.exists():continue
     result=json.loads(result_path.read_text())
     if not result.get('native_run'):continue
     extra=task_errors(Path(result['native_run']));extra['source_version']=version
     serialized=json.dumps(extra,ensure_ascii=False,indent=2)+'\n'
     (result_path.parent.parent/'additional_task_feedback.json').write_text(serialized)
     temp=out.with_suffix('.tmp');temp.write_text(serialized);temp.replace(out)
  time.sleep(.5)
