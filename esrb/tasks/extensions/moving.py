"""Industrial conveyor kitting and continuous moving-fixture assembly."""
from esrb.tasks.production.workcell import ProductionWorkcellTask
from esrb.agent.production_policy import ProductionPolicy


class MovingTransportTask(ProductionWorkcellTask):
    name = "moving_transport"
    mode = "transport"


class MovingAssemblyTask(ProductionWorkcellTask):
    name = "moving_assembly"
    mode = "moving_assembly"


class MovingTransportPolicy(ProductionPolicy):
    name = "moving_transport"


class MovingAssemblyPolicy(ProductionPolicy):
    name = "moving_assembly"
