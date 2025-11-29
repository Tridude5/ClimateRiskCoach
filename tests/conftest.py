import sys
from pathlib import Path

# Go up one directory from the test folder
project_root = Path(__file__).resolve().parents[1]

# Add to path if not already added
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))