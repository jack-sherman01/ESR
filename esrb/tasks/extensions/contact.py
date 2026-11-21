"""Panda fixture-guided pushing and precision datum alignment."""
from esrb.tasks.contact_workcells.workcell import PandaContactTask
from esrb.tasks.contact_workcells.policy import PandaContactPolicy

class ConstrainedPushTask(PandaContactTask):
    name='constrained_push'

class ForceAlignmentTask(PandaContactTask):
    name='force_alignment'

class ContactPolicy(PandaContactPolicy):
    name='contact_policy'
