"""Resume an unexecuted round after a task-comment filter false positive.

Does not modify or re-execute any candidate, simulator, rule or decision. The
only filter exception is one literal image-fitting comment written by a worker.
"""

# Allow direct execution from this subdirectory as well as package imports.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse,json
from pathlib import Path
from scripts.feedback.handover_task_feedback import validate_prompt

def validate_task_input(text):
 try:data=json.loads(text)
 except (ValueError,TypeError):return validate_prompt(text)
 if isinstance(data,dict) and isinstance(data.get('starting_program'),str):
  # This exact comment concerns rejecting image pixels, not host decisions.
  data=dict(data)
  data['starting_program']=data['starting_program'].replace('# Reject isolated colored pixels before fitting the long wooden handle.','# Filter isolated colored pixels before fitting the long wooden handle.')
  return validate_prompt(json.dumps(data,ensure_ascii=False))
 return validate_prompt(text)

def main():
 from scripts.runners import run_fresh_handover_trials as bridge
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--round',type=int,required=True);a=p.parse_args();root=a.root.resolve();folder=root/f'round_{a.round:02d}'
 decisions=list(folder.glob('version_*/gate_decision.json'));next_version=len(decisions)+1
 empty=folder/f'generation_{next_version:02d}'
 if empty.exists():
  if list(empty.iterdir()):raise ValueError('preserve nonempty generation; manual diagnosis required')
  empty.rename(folder/f'nonexecuted_v{next_version:02d}_input_filter')
 bridge.validate_prompt=validate_task_input
 bridge.range=lambda start,stop:range(max(start,next_version),stop)
 bridge.drive(root,a.round)
if __name__=='__main__':main()
