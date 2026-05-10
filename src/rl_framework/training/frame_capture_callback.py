"""Callback to capture environment frames during training for live visualization."""
from __future__ import annotations

import base64
import threading
from collections import deque
from dataclasses import dataclass
from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image
from stable_baselines3.common.callbacks import BaseCallback


@dataclass
class FrameData:
    """A single captured frame with metadata."""
    frame_index: int
    timestep: int
    episode_num: int
    image_base64: str  # JPEG base64 string


class FrameCaptureCallback(BaseCallback):
    """Captures RGB frames from env.render() at regular intervals.

    Stores frames in a bounded ring buffer to avoid memory bloat.
    Frames are encoded as JPEG base64 for efficient transmission.

    Thread-safety: _lock guards all access to self.frames and self.frame_index.
    The training thread (writes) and the API polling thread (reads) both hold
    the lock when touching the deque.
    """

    def __init__(
        self,
        capture_interval: int = 50,
        max_frames: int = 200,
        verbose: int = 0,
    ):
        """
        Args:
            capture_interval: Capture frame every N timesteps
            max_frames: Maximum frames to keep in buffer (ring buffer)
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.capture_interval = capture_interval
        self.max_frames = max_frames
        self.frames: deque[FrameData] = deque(maxlen=max_frames)
        self.frame_index = 0
        self.episode_num = 0
        self.last_capture_step = 0
        self._lock = threading.Lock()

    def _on_step(self) -> bool:
        """Called after every env.step()."""
        # Count completed episodes using the done flags SB3 provides in locals.
        dones = self.locals.get("dones")
        if dones is not None:
            self.episode_num += int(sum(dones))

        current_step = self.model.num_timesteps
        if current_step - self.last_capture_step >= self.capture_interval:
            self._capture_frame(current_step)
            self.last_capture_step = current_step

        return True

    def _capture_frame(self, timestep: int) -> None:
        """Capture a frame from the environment."""
        try:
            env = self.training_env
            if env is None:
                return

            # Try to get a frame from the environment
            frame = None

            # Try calling render() on VecEnv (newer API)
            try:
                result = env.render()
                if result is not None:
                    if isinstance(result, list) and len(result) > 0:
                        frame = result[0]
                    elif isinstance(result, np.ndarray):
                        if len(result.shape) == 4:  # Multiple envs: (n, h, w, c)
                            frame = result[0]
                        else:  # Single env: (h, w, c)
                            frame = result
            except (AttributeError, TypeError):
                # Fallback: try to get underlying envs directly
                if hasattr(env, "envs") and len(env.envs) > 0:
                    try:
                        frame = env.envs[0].render()
                    except Exception:
                        pass

            if frame is None or not isinstance(frame, np.ndarray):
                return

            # Ensure frame is uint8 RGB
            if frame.dtype != np.uint8:
                frame = np.uint8(np.clip(frame, 0, 255))

            if frame.shape[2] == 4:  # RGBA, drop alpha
                frame = frame[:, :, :3]

            # Encode frame as JPEG base64
            img = Image.fromarray(frame, mode="RGB")
            buffer = BytesIO()
            img.save(buffer, format="JPEG", quality=85, optimize=False)
            img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

            # Create frame data and add to buffer under lock so the API
            # polling thread sees a consistent snapshot.
            with self._lock:
                frame_data = FrameData(
                    frame_index=self.frame_index,
                    timestep=timestep,
                    episode_num=self.episode_num,
                    image_base64=img_base64,
                )
                self.frames.append(frame_data)
                self.frame_index += 1

            if self.verbose >= 2:
                print(f"[FrameCapture] Captured frame {self.frame_index} at step {timestep}")
        except Exception as e:
            if self.verbose >= 1:
                print(f"[FrameCapture] Error capturing frame: {e}")

    def get_frames(self, since: int = 0) -> list[dict[str, Any]]:
        """Return captured frames as dict list for API.

        Args:
            since: Only return frames with frame_index >= this value.
                   Callers pass the last seen frame_index + 1 so they only
                   receive frames they haven't processed yet.
        """
        with self._lock:
            return [
                {
                    "frame_index": f.frame_index,
                    "timestep": f.timestep,
                    "episode_num": f.episode_num,
                    "image_base64": f.image_base64,
                }
                for f in self.frames
                if f.frame_index >= since
            ]
