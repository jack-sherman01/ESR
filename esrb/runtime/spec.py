"""Lightweight scene capabilities injected into the common runtime."""
from dataclasses import asdict, dataclass, field
from importlib import import_module


@dataclass(frozen=True)
class SceneSpec:
    key: str
    prefix: str
    task: str
    environment: str
    prompt: str
    safety_module: str
    chain_port: int
    chain_gpu: int
    protocol_defaults: dict = field(default_factory=dict)
    input_module: str | None = None

    @property
    def rules_path(self):
        return f'configs/experiment/{self.prefix}_safety_gate.json'

    @property
    def api_path(self):
        return f'integrations/aspire/{self.prefix}_skill_api.txt'

    def validate_rules(self, rules):
        return import_module(self.safety_module).validate_rules(rules)

    def decide(self, *args, **kwargs):
        module = self.safety_module if self.prefix == 'wipe' else 'esrb.detection.admission'
        return import_module(module).decide(*args, **kwargs)

    def validate_prompt(self, text):
        if self.input_module:
            import_module(self.input_module).validate_prompt(text)

    def archive_inputs(self, folder, base, previous):
        if self.input_module:
            return import_module(self.input_module).archive_inputs(folder, base, previous)

    def feedback(self, folder):
        from .feedback import feedback
        from .io import read
        if self.prefix == 'handover':
            return import_module(self.input_module).feedback(folder)
        result = feedback(folder)
        if self.prefix == 'wipe':
            last = read(folder / 'execution/result.json').get('last_telemetry') or {}
            if type(last.get('wiped_markers')) is int and type(last.get('num_markers')) is int:
                result['task_progress'] = {'cleaned_markers': last['wiped_markers'], 'total_markers': last['num_markers']}
        return result

    def document(self):
        return asdict(self)


def scene_from_protocol(protocol):
    if protocol.get('execution_layout') != 'unified-cli/1' or 'scene' not in protocol:
        raise ValueError('Historical execution requires its frozen source checkout; create a new batch for the unified CLI')
    return SceneSpec(**protocol['scene'])
