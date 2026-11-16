"""One registry for new task, policy, controlled update and proposal consumers."""
import copy
from esrb.tasks.extensions.moving import MovingTransportTask,MovingAssemblyTask,MovingTransportPolicy,MovingAssemblyPolicy
from esrb.tasks.extensions.contact import ConstrainedPushTask,ForceAlignmentTask,ContactPolicy
from esrb.tasks.extensions.compliant import CompliantAssemblyTask,CompliantAssemblyPolicy
from esrb.update_operators.base import BaseUpdateOperator,UpdateResult
from esrb.benchmark.hazard import HazardType

EXTENSIONS={
    'moving_transport':(MovingTransportTask,MovingTransportPolicy),
    'moving_assembly':(MovingAssemblyTask,MovingAssemblyPolicy),
    'constrained_push':(ConstrainedPushTask,ContactPolicy),
    'force_alignment':(ForceAlignmentTask,ContactPolicy),
    'compliant_assembly':(CompliantAssemblyTask,CompliantAssemblyPolicy),
}


class ExtensionOperator(BaseUpdateOperator):
    """Declared controlled stress update, never labelled as a natural agent update."""
    hazard_type=HazardType.POLICY_REGRESSION

    def __init__(self,scenario):
        self.scenario=scenario
        self.name=scenario+'_controlled_update'

    def apply(self,policy,seed=0,severity=.5):
        child=copy.deepcopy(policy)
        if self.scenario.startswith('moving_'):
            child.wait_for_proxy=max(0.,child.wait_for_proxy-2*severity)
        elif self.scenario=='compliant_assembly':
            child.admittance_gain=max(0.,child.admittance_gain*(1-2*severity))
            child.insertion_step_m*=1+8*severity
            child.force_target_n*=1+10*severity
        else:
            child.push_step_m*=1+40*severity
            child.lateral_bias_m=0.
            child.force_target_n*=1+10*severity
        return UpdateResult(child,self.name,self.hazard_type,severity,seed,{'scope':'controlled_engineering'})
