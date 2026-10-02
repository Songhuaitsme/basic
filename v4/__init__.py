"""V4 net-benefit-gated active-wait runtime."""

SYSTEM_VERSION = "4.0"

from .gating import NetBenefitWaitGate, WaitGateConfig
from .v4_runtime import create_v4_runtime

__all__ = [
    "SYSTEM_VERSION",
    "NetBenefitWaitGate",
    "WaitGateConfig",
    "create_v4_runtime",
]
