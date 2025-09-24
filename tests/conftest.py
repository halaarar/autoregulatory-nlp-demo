"""
Pytest configuration that ensures the project root is on sys.path
so test modules can import run_demo and other top-level files.
"""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
