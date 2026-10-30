"""Live Snakebot launcher using 100-frame bursts and 3x servo angle scaling."""

from dataclasses import dataclass

import tyro

import mjlab
from mjlab.scripts.snakebot_live import LiveSnakebotConfig, run_live


@dataclass(frozen=True)
class BurstSnakebotConfig(LiveSnakebotConfig):
  burst_size: int = 100
  """Complete ten-angle frames per batch; default playback lasts five seconds."""


def main() -> None:
  run_live(tyro.cli(BurstSnakebotConfig, config=mjlab.TYRO_FLAGS))


if __name__ == "__main__":
  main()
