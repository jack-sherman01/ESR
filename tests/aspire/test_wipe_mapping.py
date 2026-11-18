"""Provider offset must resolve the requested wiping plane, in any orientation."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from scripts.adapters.wipe_tool_mapping import provider_input_transform


def test_calibrated_wiping_plane_roundtrip_at_varied_orientation():
    local=np.array([[0.,1.,0.,0.],[-1.,0.,0.,0.],[0.,0.,1.,.03],[0.,0.,0.,1.]])
    for angle in [0.,.4,1.6]:
        c,s=np.cos(angle),np.sin(angle)
        target=np.array([[c,0.,s,.5],[0.,1.,0.,-.2],[-s,0.,c,.9],[0.,0.,0.,1.]])
        payload=provider_input_transform(target,local)
        solved_hand=payload.copy()
        solved_hand[:3,3]+=solved_hand[:3,:3]@np.array([0.,0.,-.1])
        np.testing.assert_allclose(solved_hand@local,target,atol=1e-14)
