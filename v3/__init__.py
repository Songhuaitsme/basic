"""V3 candidate-generation performance runtime.

V3 reuses the frozen V1 domain, scheduler, feature, and model contracts.  Its
changes are isolated to this package and are injected only by the V3 entry
points.
"""

SYSTEM_VERSION = "3.0"

from .v3_runtime import create_v3_runtime

__all__ = ["SYSTEM_VERSION", "create_v3_runtime"]
