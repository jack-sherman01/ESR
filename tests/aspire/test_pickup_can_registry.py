"""The process-local registration adds only the unchanged native Can task."""
import pytest
pytest.importorskip('cap_harness', reason='requires the optional ASPIRE runtime')
from cap_harness.robosuite import registry
from esrb.backends.aspire.pickup_can_registry import install_registry


def test_registration_is_idempotent_and_keeps_native_task_metadata(monkeypatch):
    original=registry.RobosuiteTaskRegistry
    before=original().enumerate_tasks()
    monkeypatch.setattr(registry,'RobosuiteTaskRegistry',original)
    first=install_registry()
    installed=registry.RobosuiteTaskRegistry
    assert install_registry()==first
    assert registry.RobosuiteTaskRegistry is installed
    tasks=installed().enumerate_tasks()
    assert tasks[:-1]==before
    assert len(tasks)==len(before)+1
    assert first.task_name=='pickup_can' and first.environment=='PickPlaceCan'
    assert first.robots==('Panda',)
