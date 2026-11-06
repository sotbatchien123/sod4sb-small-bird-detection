"""python -u scripts/run_qos.py scripts/<script>.py [args]: run a script with Windows EcoQoS disabled.

Kept outside the trainers so their implementation hashes (checked on resume) do not change.
"""
import runpy, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from win_qos import disable_power_throttling

disable_power_throttling()
script = sys.argv[1]
sys.argv = sys.argv[1:]
sys.path[0] = str(Path(script).resolve().parent)
runpy.run_path(script, run_name='__main__')
