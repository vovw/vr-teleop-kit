"""Optional-lerobot compatibility layer.

``lerobot`` (and the torch stack it drags in) is only needed when these
adapters plug into the LeRobot ecosystem — ``lerobot-record``, dataset
tooling, policy rollouts. A robot box that only teleoperates gets no
value from a multi-GB torch install, so when ``lerobot`` is absent the
adapters fall back to minimal stand-ins with the same surface:

  Teleoperator / TeleoperatorConfig — base class + config dataclass with
      the fields the adapters actually use (``id``, ``calibration_dir``)
      and a no-op ``register_subclass`` (plugin registration only means
      something inside LeRobot CLIs).
  init_logging — plain ``logging.basicConfig`` stand-in.

``HAS_LEROBOT`` tells callers which world they are in.
"""

from __future__ import annotations

try:
    from lerobot.teleoperators.teleoperator import (  # noqa: F401
        Teleoperator,
        TeleoperatorConfig,
    )
    from lerobot.utils.utils import init_logging  # noqa: F401

    HAS_LEROBOT = True
except ImportError:
    HAS_LEROBOT = False

    import abc
    import logging
    from dataclasses import dataclass
    from pathlib import Path

    @dataclass
    class TeleoperatorConfig:  # type: ignore[no-redef]
        id: str | None = None
        calibration_dir: Path | None = None

        @classmethod
        def register_subclass(cls, name: str):
            """Stand-in for draccus' ChoiceRegistry decorator — plugin
            registration only exists inside LeRobot CLIs."""

            def _wrap(subclass):
                return subclass

            return _wrap

    class Teleoperator(abc.ABC):  # type: ignore[no-redef]
        """Minimal stand-in for LeRobot's Teleoperator base class: keeps
        the config identity, skips the calibration-file machinery (the
        Quest teleop needs none — ``is_calibrated`` is always True)."""

        name: str

        def __init__(self, config: TeleoperatorConfig) -> None:
            self.id = config.id
            self.calibration_dir = (
                Path(config.calibration_dir) if config.calibration_dir else None
            )

    def init_logging(*args, **kwargs) -> None:  # type: ignore[no-redef]
        logging.basicConfig(
            level=logging.INFO,
            format="%(levelname)s %(asctime)s %(name)s:%(lineno)d %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
