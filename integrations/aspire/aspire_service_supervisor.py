#!/usr/bin/env python3
"""Foreground supervisor for local ASPIRE provider launchers.

No GPT service or API key is required. --dry-run reads local paths only; it
neither creates output nor probes/reserves ports nor launches a subprocess.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request

PORTS = (8114, 8115, 8116)


def build_plan(runtime: Path, receipt: Path, output: Path, sam3_precision: str) -> list[dict]:
    runtime = runtime.resolve()
    scripts = Path(__file__).resolve().parent
    vendor = runtime / 'third_party/contact_graspnet_pytorch'
    cache = Path.home() / '.cache/robot_descriptions'
    required = [receipt, vendor / 'checkpoints/contact_graspnet/checkpoints/model.pt',
                cache / 'example-robot-data/robots/panda_description/urdf/panda.urdf']
    specs = [('sam3', '.venv-sam3', 'aspire_sam3_service.py',
              ['--receipt', str(receipt), '--startup-report', str(output / 'sam3-startup.json'),
               '--precision', sam3_precision, '--retain-public-topk', '--device', 'cuda:0', '--port', '8114']),
             ('graspnet', '.venv-graspnet', 'aspire_graspnet_cpu_service.py',
              ['--vendor', str(vendor), '--startup-report', str(output / 'graspnet-startup.json')]),
             ('pyroki', '.venv-pyroki', 'aspire_pyroki_service.py',
              ['--out', str(output / 'pyroki-startup.json'), '--port', '8116'])]
    for _, env, script, _ in specs:
        required.extend([runtime / env / 'bin/python', scripts / script])
    for path in required:
        if not path.is_file():
            raise ValueError(f'required local file missing: {path}')
    data = json.loads(receipt.read_text())
    for info in data['files'].values():
        if not Path(info['path']).is_file():
            raise ValueError(f'local SAM3 receipt asset missing: {info["path"]}')
    # Existing cache is mandatory; Git networking is disabled in child env too.
    environment = {'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                   'GIT_ALLOW_PROTOCOL': 'file', 'GIT_TERMINAL_PROMPT': '0',
                   'ROBOT_DESCRIPTIONS_CACHE': str(cache), 'PYTHONUNBUFFERED': '1'}
    result = []
    for (name, env, script, args), port in zip(specs, PORTS, strict=True):
        python = runtime / env / 'bin/python'
        if not os.access(python, os.X_OK):
            raise ValueError(f'interpreter is not executable: {python}')
        overrides = dict(environment)
        if name in {'graspnet', 'pyroki'}:
            overrides.update(CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu')
        else:
            # cuda:0 is relative to the caller's existing visible devices.
            overrides.update(PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
        result.append({'name': name, 'port': port, 'command': [str(python), str(scripts / script), *args],
                       'cwd': str(runtime), 'environment_overrides': overrides,
                       'launcher_sha256': hashlib.sha256((scripts / script).read_bytes()).hexdigest(),
                       'log': str(output / f'{name}.log')})
    return result


def check_ports(ports=PORTS) -> None:
    held = []
    try:
        for port in ports:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            held.append(sock)
            try:
                sock.bind(('127.0.0.1', port))
            except OSError as exc:
                raise RuntimeError(f'port {port} unavailable; stop its existing owner first') from exc
    finally:
        for sock in held:
            sock.close()


def health(port: int) -> bool:
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(f'http://127.0.0.1:{port}/healthz', timeout=1) as response:
            return response.status == 200 and json.load(response).get('ok') is True
    except (OSError, ValueError, AttributeError):
        return False


def child_environment(overrides: dict) -> dict:
    env = {k: v for k, v in os.environ.items()
           if not (k.endswith(('_API_KEY', '_TOKEN')) or k in {'PYTHONPATH', 'PYTHONHOME', 'ROBOT_DESCRIPTION_COMMIT'})}
    env.update(overrides)
    return env


def assert_alive(children: list) -> None:
    for spec, process in children:
        code = process.poll()
        if code is not None:
            raise RuntimeError(f'{spec["name"]} unexpectedly exited ({code}); see {spec["log"]}')


def cleanup(children: list, timeout: float) -> None:
    # Each Popen creates its own session; signal only those owned groups, including
    # descendants if a provider process has already exited.
    for _, process in children:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + timeout
    for _, process in children:
        try:
            process.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            pass
    for _, process in children:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    for _, process in children:
        process.wait()


def supervise(plan: list, output: Path, ready_timeout: float, shutdown_timeout: float) -> int:
    check_ports(tuple(spec['port'] for spec in plan))
    output.mkdir(parents=True, exist_ok=False)
    children, logs = [], []
    stop = {'signal': None}
    receipt = {'schema_version': 1, 'status': 'starting', 'services': plan,
               'started_unix_s': time.time(), 'gpt_api_called': False,
               'parent_pid': os.getpid(), 'launcher_script': str(Path(__file__).resolve()),
               'parent_start_ticks': int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19])}
    def save():
        tmp = output / 'supervisor.json.tmp'
        tmp.write_text(json.dumps(receipt, indent=2) + '\n')
        tmp.replace(output / 'supervisor.json')
    def interrupted(signum, frame):
        stop['signal'] = signum
    handlers = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM)}
    code = 0
    try:
        save()
        for spec in plan:
            if stop['signal'] is not None:
                break
            log = Path(spec['log']).open('x')
            logs.append(log)
            proc = subprocess.Popen(spec['command'], cwd=spec['cwd'],
                                    env=child_environment(spec['environment_overrides']),
                                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            children.append((spec, proc))
            spec['pid'] = proc.pid
            save()
            deadline = time.monotonic() + ready_timeout
            while stop['signal'] is None:
                assert_alive(children)
                if health(spec['port']):
                    spec['ready_unix_s'] = time.time()
                    save()
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError(f'{spec["name"]} readiness timeout; see {spec["log"]}')
                time.sleep(.25)
        if stop['signal'] is None:
            receipt['status'] = 'ready'
            save()
            ports = '/'.join(str(spec['port']) for spec in plan)
            print(f'ASPIRE services ready on {ports}; logs: {output}; Ctrl+C stops all.', flush=True)
        while stop['signal'] is None:
            assert_alive(children)
            time.sleep(.25)
    except Exception as exc:
        code = 1
        receipt['error'] = f'{type(exc).__name__}: {exc}'
        print(receipt['error'], file=sys.stderr, flush=True)
    finally:
        receipt['status'] = 'stopping'
        try:
            save()
        finally:
            cleanup(children, shutdown_timeout)
        for log in logs:
            log.close()
        receipt.update(status='failed' if code else 'stopped', stopped_unix_s=time.time(),
                       stop_signal=stop['signal'], exit_codes={s['name']: p.returncode for s, p in children})
        save()
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    return code if stop['signal'] is None else 128 + stop['signal']


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, help='new directory; default creates a unique UTC timestamp/PID name')
    parser.add_argument('--readiness-timeout', type=float, default=300, help='seconds per service')
    parser.add_argument('--shutdown-timeout', type=float, default=10)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--sam3-precision', choices=['bf16-backbone','float32-offload','float32'], default='bf16-backbone')
    args = parser.parse_args(argv)
    if (not math.isfinite(args.readiness_timeout) or not math.isfinite(args.shutdown_timeout)
            or args.readiness_timeout <= 0 or args.shutdown_timeout < 0):
        parser.error('timeouts must be positive (shutdown may be zero)')
    output = (args.output_dir or args.checkout / 'outputs' /
              f'services-{time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())}-{os.getpid()}').resolve()
    try:
        if output.exists():
            raise ValueError(f'output directory already exists: {output}')
        plan = build_plan(args.checkout, args.receipt.resolve(), output, args.sam3_precision)
        if args.dry_run:
            print(json.dumps({'dry_run': True, 'ports_checked': False, 'output_dir': str(output), 'services': plan}, indent=2))
            return 0
        return supervise(plan, output, args.readiness_timeout, args.shutdown_timeout)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.exit(1, f'{exc}\n')

if __name__ == '__main__':
    raise SystemExit(main())
