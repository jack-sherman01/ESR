"""Keep the actual coordinator exit status, including interrupted execution."""
from datetime import datetime, timezone
import subprocess
from .io import save


def supervise(command, root, *, cwd=None):
    with (root / 'driver.log').open('a') as log:
        process = subprocess.Popen(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT)
        try:
            returncode = process.wait()
        except KeyboardInterrupt:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            returncode = 130
        except BaseException:
            process.terminate()
            process.wait()
            raise
        finally:
            save(root / 'driver_exit.json', {'returncode': locals().get('returncode', process.returncode),
                 'finished_at_utc': datetime.now(timezone.utc).isoformat()})
    return returncode
