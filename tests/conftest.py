import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

T0 = datetime(2026, 10, 28, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def t0():
    return T0
