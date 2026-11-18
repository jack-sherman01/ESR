"""Copy only standard Python/API validation errors to a worker's own trial directory."""

# Allow direct execution from this subdirectory as well as package imports.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pathlib import Path
import hashlib,json,re,time
ROOT=Path('/home/zhifeng/ESR/.esr-local/fresh-can-wipe-20261117')
BLOCKED=re.compile(r'esrb|\bsafety\b|\bgate\b|violation|interlock|forbidden',re.I)
ALLOWED={'NameError','ValueError','TypeError','AttributeError','IndexError','KeyError','ZeroDivisionError'}

def extract(native):
 path=native/'source/program-result.json'
 if not path.exists():return None
 raw=json.loads(path.read_text());error=raw.get('error')
 if not isinstance(error,dict):return None
 message=error.get('message','');kind=(error.get('details') or {}).get('exception_type')
 if BLOCKED.search(message):return None
 if error.get('code')=='invalid_request' and message.startswith('program rejected:'):
  return {'program_api_error':message.removeprefix('program rejected:').strip()[:350]}
 if kind in ALLOWED and error.get('code')=='execution_failed':
  return {'python_error':{'type':kind,'message':message[:350]}}
 return None

if __name__=='__main__':
 while True:
  for scene in ['pickup_can','wipe-formal']:
   for assignment in (ROOT/scene).glob('round_*/agent_assignment.json'):
    config=json.loads(assignment.read_text());capsule=Path(config['capsule'])
    for result_path in assignment.parent.glob('version_*/execution/result.json'):
     version=int(result_path.parent.parent.name[-2:]);out=capsule/f'additional_feedback_v{version:02d}.json'
     if out.exists():continue
     result=json.loads(result_path.read_text())
     if not result.get('native_run'):continue
     extra=extract(Path(result['native_run']))
     if extra is None:extra={'program_api_error':None}
     extra['source_version']=version
     archive=result_path.parent.parent/'additional_task_feedback.json'
     serialized=json.dumps(extra,ensure_ascii=False,indent=2)+'\n'
     archive.write_text(serialized);temp=out.with_suffix('.tmp');temp.write_text(serialized);temp.replace(out)
  time.sleep(.5)
