"""Process-local ASPIRE registration of the unchanged native PickPlaceCan task."""
def install_registry():
    from cap_harness.robosuite import registry as module
    original=module.RobosuiteTaskRegistry
    metadata=module.RobosuiteTaskMetadata('pickup_can','PickPlaceCan',
        'Pick up the soda can and place it into the bin compartment marked by the can silhouette, then release and move the gripper away.',
        ('Panda',),('agentview','robot0_eye_in_hand'))
    class Registry(original):
        def __init__(self,tasks=None):
            super().__init__(tasks if tasks is not None else module.ROBOSUITE_TASKS+(metadata,))
    module.RobosuiteTaskRegistry=Registry
    return metadata
