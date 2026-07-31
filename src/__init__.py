"""
Initialization module for search relevancy package.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

__version__ = "1.0.0"
__author__ = "Search Relevancy Team"
