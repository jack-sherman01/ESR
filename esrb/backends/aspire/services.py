"""Configure explicitly provisioned provider services; never start them implicitly."""
from functools import partial


def configure_native_runtime(run, protocol):
    run.PyRokiProvider = partial(run.PyRokiProvider, base_url=protocol['runtime']['ik_service_url'])
