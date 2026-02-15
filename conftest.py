"""Root conftest.py to configure pytest import paths."""

import sys
from pathlib import Path

# Add src/ directory to Python path for imports
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))
