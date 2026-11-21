"""Tool-free ephemeral calls with explicit, chain-local model inputs."""
import json
from pathlib import Path
import shutil
import subprocess

from .io import read, save, digest
from .process import child_env, run_logged

DISABLED = ('shell_tool', 'unified_exec', 'multi_agent', 'multi_agent_v2', 'apps',
            'view_image', 'image_generation', 'browser_use', 'computer_use', 'goals',
            'skill_search', 'external_agent_memory_import', 'code_mode', 'code_mode_host')
SCHEMA = {'type': 'object', 'properties': {'program': {'type': 'string'},
          'rationale': {'type': 'string'}}, 'required': ['program', 'rationale'],
          'additionalProperties': False}


class ModelBlocked(RuntimeError):
    pass


def command(binary, workspace, output, model):
    args = [binary, 'exec', '--json', '--ephemeral', '--ignore-user-config', '--ignore-rules',
            '--skip-git-repo-check', '--sandbox', 'read-only', '--enable', 'skip_host_skill_discovery']
    for feature in DISABLED:
        args += ['--disable', feature]
    args += ['-c', 'approval_policy="never"', '-c', 'web_search="disabled"',
             '-c', 'project_doc_max_bytes=0', '-c', 'model_reasoning_effort="medium"',
             '-c', 'developer_instructions="Generate the requested Python skill as JSON. No tools."',
             '--model', model, '--cd', str(workspace), '--output-schema', str(output / 'schema.json'),
             '--output-last-message', str(output / 'response.json'), '-']
    return args


def generate(output, prompt, model='gpt-6.1-sol', binary=None, timeout=900):
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    workspace = output / 'empty_workspace'; workspace.mkdir()
    save(output / 'schema.json', SCHEMA)
    (output / 'prompt.txt').write_text(prompt)
    binary = binary or shutil.which('codex')
    if not binary:
        raise ModelBlocked('Codex executable unavailable')
    save(output / 'generation_config.json', {'model': model, 'reasoning_effort': 'medium',
         'history': json.loads(prompt).get('context_policy', 'legacy full chain-local transcript'),
         'prompt_chars': len(prompt), 'prompt_utf8_bytes': len(prompt.encode()),
         'tools_disabled': list(DISABLED), 'project_doc_max_bytes': 0,
         'binary': binary, 'binary_sha256': digest(Path(binary).resolve())})
    try:
        run_logged(command(binary, workspace, output, model), workspace, output, env=child_env(),
                   timeout=timeout, stdin=prompt.encode(), stdout='events.jsonl')
    except (RuntimeError, OSError, subprocess.TimeoutExpired) as exc:
        raise ModelBlocked(f'model call incomplete: {exc}; inspect {output}') from exc
    events = [json.loads(line) for line in (output / 'events.jsonl').read_text().splitlines()
              if line.strip().startswith('{')]
    for event in events:
        item = event.get('item') or {}
        if item.get('type') in {'command_execution', 'mcp_tool_call', 'web_search', 'file_change',
                                'image_generation', 'collab_agent_tool_call'}:
            raise ModelBlocked('generation isolation violated: a tool was used')
    answer = read(output / 'response.json')
    if set(answer) != {'program', 'rationale'} or not all(isinstance(v, str) for v in answer.values()):
        raise ValueError('invalid structured model answer')
    save(output / 'model_receipt.json', {'requested_model': model, 'automatic_fallback': False,
         'threads': [e for e in events if e.get('type') == 'thread.started'],
         'completion': [e for e in events if e.get('type') == 'turn.completed'],
         'prompt_sha256': digest(output / 'prompt.txt'), 'response_sha256': digest(output / 'response.json')})
    return answer
