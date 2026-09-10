import sys
from pathlib import Path

import pytest

# 项目根（tests/smoke -> 上两级 = my_agent），确保 `import src...` 可用
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SAMPLE_CONFIG = PROJECT_ROOT / "tests" / "smoke" / "sample_config.yaml"


@pytest.fixture(scope="session")
def project_root():
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def sample_config_path():
    return SAMPLE_CONFIG
