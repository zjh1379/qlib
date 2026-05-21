import sys
from pathlib import Path

# Ensure `app` is importable from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
