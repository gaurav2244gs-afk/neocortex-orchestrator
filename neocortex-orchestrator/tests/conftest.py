import sys
from pathlib import Path

# Allow `import neocortex...` without an editable install during testing.
SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
