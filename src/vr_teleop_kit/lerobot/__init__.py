"""LeRobot ``Teleoperator`` adapters.

When ``lerobot`` is installed, importing the submodules registers the
config types with LeRobot's ``TeleoperatorConfig`` registry, after which
``--teleop.type=bi_quest_teleop`` / ``single_arm_quest_teleop`` work in
LeRobot CLIs. Without ``lerobot`` the same classes still work standalone
(teleop loops, the examples) via lightweight stand-in base classes —
no torch required. ``HAS_LEROBOT`` reports which mode is active.
"""

from ._compat import HAS_LEROBOT, init_logging
from .bi_quest_teleop import BiQuestTeleoperator, BiQuestTeleoperatorConfig
from .single_arm_quest_teleop import (
    SingleArmQuestTeleoperator,
    SingleArmQuestTeleoperatorConfig,
)

__all__ = [
    "BiQuestTeleoperator",
    "BiQuestTeleoperatorConfig",
    "SingleArmQuestTeleoperator",
    "SingleArmQuestTeleoperatorConfig",
    "HAS_LEROBOT",
    "init_logging",
]
