"""
Shared pytest configuration.

Sizes of the plan's property tests (§2.3):
    default             fast local runs (2,000 random playouts)
    CHATURANGA_FULL_TESTS=1   the plan's full sizes (10,000) — CI sets this
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

FULL_TESTS = os.environ.get('CHATURANGA_FULL_TESTS') == '1'
N_PLAYOUTS = 10_000 if FULL_TESTS else 2_000
N_POSITIONS = 10_000
