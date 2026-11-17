"""CPU placement for upstream Contact-GraspNet; local trusted checkpoint only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch
import uvicorn
from cap_harness.providers.graspnet import service

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--vendor', type=Path, required=True)
parser.add_argument('--startup-report', type=Path, required=True)
args = parser.parse_args()
if args.startup_report.exists():
    parser.error('Report exists')
vendor = args.vendor.resolve()
sys.path.extend([str(vendor/'Pointnet_Pointnet2_pytorch'), str(vendor)])
from contact_graspnet_pytorch.contact_grasp_estimator import GraspEstimator
from contact_graspnet_pytorch.checkpoints import CheckpointIO
checkpoint_root = vendor/'checkpoints/contact_graspnet'
checkpoint = checkpoint_root/'checkpoints/model.pt'
torch.set_num_threads(4)
estimator = GraspEstimator(service._load_config(checkpoint_root))
assert estimator.device.type == 'cpu'
checkpoint_io = CheckpointIO(checkpoint_dir=checkpoint_root/'checkpoints', model=estimator.model)
checkpoint_io.parse_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False))
service._ESTIMATOR = estimator
report = {'status':'loaded','device':'cpu','checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
          'torch':torch.__version__,'parameter_count':sum(p.numel() for p in estimator.model.parameters()),
          'adaptation':'checkpoint map_location=cpu; inference handlers unchanged','gpt_api_called':False}
args.startup_report.parent.mkdir(parents=True,exist_ok=True)
with args.startup_report.open('x') as handle: json.dump(report,handle,indent=2)
print(json.dumps(report),flush=True)
uvicorn.run(service.app,host='127.0.0.1',port=8115)
