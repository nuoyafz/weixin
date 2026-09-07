import sys
import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_SRC_ROOT = Path(__file__).resolve().parent

if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))
