"""Panda horizontal keyed-shaft insertion with contact admittance."""
from esrb.tasks.contact_workcells.workcell import PandaContactTask
from esrb.tasks.contact_workcells.policy import PandaContactPolicy

class CompliantAssemblyTask(PandaContactTask):
    name='compliant_assembly'

class CompliantAssemblyPolicy(PandaContactPolicy):
    name='compliant_assembly'
