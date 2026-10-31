"""Read a Windows XInput right stick, including from a WSL simulation process."""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Callable


def right_stick_direction(rx: int, ry: int, deadzone: float) -> tuple[float, float]:
  """Return forward/left strengths, with a radial dead zone and unit max speed."""
  x = max(-1.0, min(1.0, rx / 32767.0))
  y = max(-1.0, min(1.0, ry / 32767.0))
  radius = math.hypot(x, y)
  if radius <= deadzone:
    return 0.0, 0.0
  strength = (min(radius, 1.0) - deadzone) / (1.0 - deadzone)
  # XInput +RY is up/forward; +RX is right, opposite our positive left axis.
  return y / radius * strength, -x / radius * strength


def gamepad_command(index: int) -> list[str]:
  helper = Path(__file__).with_suffix(".ps1")
  executable = shutil.which("powershell.exe")
  if executable is None:
    raise RuntimeError("Windows PowerShell is unavailable for the XInput reader")
  if os.name == "nt":
    helper_path = str(helper)
  else:
    try:
      helper_path = subprocess.run(
        ["wslpath", "-w", str(helper)],
        capture_output=True,
        text=True,
        check=True,
        timeout=5,
      ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
      raise RuntimeError(
        "The XInput reader needs Windows or Windows interop in WSL"
      ) from exc
  return [
    executable,
    "-NoProfile",
    "-NonInteractive",
    "-WindowStyle",
    "Hidden",
    "-ExecutionPolicy",
    "Bypass",
    "-File",
    helper_path,
    "-ControllerIndex",
    str(index),
  ]


class RightStickReader:
  """Keep only the latest sample; stale/disconnected readers produce no movement."""

  def __init__(
    self,
    index: int = -1,
    stale_timeout: float = 0.25,
    clock: Callable[[], float] = time.monotonic,
  ):
    self.index = index
    self.stale_timeout = stale_timeout
    self._clock = clock
    self._lock = Lock()
    self._closing = Event()
    self._sample: tuple[int, int, float] | None = None
    self._connected: bool | None = None
    self._process: subprocess.Popen[str] | None = None
    self._reader: Thread | None = None

  def ingest(self, line: str) -> None:
    try:
      data = json.loads(line)
    except (ValueError, UnicodeDecodeError):
      return
    if not isinstance(data, dict) or type(data.get("connected")) is not bool:
      return
    connected = data["connected"]
    if connected and any(
      type(data.get(axis)) is not int or not -32768 <= data[axis] <= 32767
      for axis in ("rx", "ry")
    ):
      return
    with self._lock:
      self._sample = (data["rx"], data["ry"], self._clock()) if connected else None
      changed = self._connected != connected
      self._connected = connected
    if changed:
      if connected:
        print(
          f"Gamepad connected: XInput slot {data.get('index')}; right stick active."
        )
      else:
        print("Gamepad disconnected/not detected; WASD remains available.")

  def read(self) -> tuple[int, int] | None:
    with self._lock:
      sample = self._sample
    if sample is None or self._clock() - sample[2] > self.stale_timeout:
      return None
    return sample[0], sample[1]

  def start(self) -> None:
    if self._process is not None:
      raise RuntimeError("Gamepad reader already started")
    self._process = subprocess.Popen(
      gamepad_command(self.index),
      stdin=subprocess.PIPE,
      stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT,
      text=True,
      bufsize=1,
      creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
      if os.name == "nt"
      else 0,
    )
    process = self._process

    def receive():
      assert process.stdout is not None
      try:
        for line in process.stdout:
          self.ingest(line)
      finally:
        with self._lock:
          self._sample = None
        if not self._closing.is_set():
          print("Gamepad reader stopped; WASD remains available.")

    self._reader = Thread(target=receive, name="snakebot-gamepad", daemon=True)
    self._reader.start()

  def close(self) -> None:
    self._closing.set()
    process = self._process
    if process is None:
      return
    if process.stdin is not None:
      try:
        process.stdin.write("stop\n")
        process.stdin.flush()
      except (OSError, ValueError):
        pass
    try:
      process.wait(timeout=2)
    except subprocess.TimeoutExpired:
      process.terminate()
      try:
        process.wait(timeout=2)
      except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=2)
    if self._reader is not None:
      self._reader.join(timeout=2)
    if process.stdin is not None:
      try:
        process.stdin.close()
      except OSError:
        pass
    if process.stdout is not None:
      process.stdout.close()
    with self._lock:
      self._sample = None
