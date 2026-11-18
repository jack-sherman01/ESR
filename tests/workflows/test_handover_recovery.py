import json,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from scripts.recovery.resume_fresh_handover_round import validate_task_input

def test_literal_image_comment_is_valid():
 validate_task_input(json.dumps({'task_prompt':'complete handover','starting_program':'# Reject isolated colored pixels before fitting the long wooden handle.\nx = 1'}))
@pytest.mark.parametrize('data',[
 {'task_prompt':'gate result','starting_program':'# Reject isolated colored pixels before fitting the long wooden handle.'},
 {'task_prompt':'complete handover','starting_program':'# reject candidate based on host result'},
 {'task_prompt':'complete handover','starting_program':None,'previous_task_feedback':{'ESR':'secret'}},
])
def test_exception_does_not_weaken_other_inputs(data):
 with pytest.raises(ValueError):validate_task_input(json.dumps(data))
