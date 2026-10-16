import os
import sys
import tempfile
from pathlib import Path

# Must happen before forge.config is imported anywhere.
os.environ["FORGE_HOME"] = tempfile.mkdtemp(prefix="forge_test_")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
