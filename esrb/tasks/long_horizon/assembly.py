"""Three-part articulated Panda assembly in one continuous episode."""
from esrb.tasks.production.workcell import ProductionWorkcellTask


class LongHorizonAssemblyTask(ProductionWorkcellTask):
    name = "long_horizon_assembly"
    mode = "assembly"
