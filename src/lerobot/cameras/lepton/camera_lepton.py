import logging
import math
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from lerobot.cameras.camera import Camera
from lerobot.cameras.lepton.configuration_lepton import LeptonCameraConfig
from lerobot.utils.errors import DeviceNotConnectedError

logger = logging.getLogger(__name__)


class LeptonCamera(Camera):
    """
    Thermal camera support for FLIR Lepton via PureThermal board.
    
    This implementation uses `v4l2-ctl` to capture raw Y16 (16-bit radiometric) data,
    which is standard for these devices on Linux. Reading directly via OpenCV often fails
    to capture the full 16-bit dynamic range correctly without specific drivers.
    """
    
    def __init__(self, config: LeptonCameraConfig):
        super().__init__(config)
        self.config = config
        self.fps = config.fps
        self.width = config.width
        self.height = config.height
        self.device_path = config.device_path
        self.colormap = config.colormap

        self.proc = None
        self.thread = None
        self.stop_event = threading.Event()
        self.frame_buffer = None
        self.frame_lock = threading.Lock()
        self._is_connected = False
        
        # Check dependencies
        if shutil.which("v4l2-ctl") is None:
            raise RuntimeError("`v4l2-ctl` not found. Please install v4l-utils (sudo apt install v4l-utils).")


    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @is_connected.setter
    def is_connected(self, value: bool):
        self._is_connected = value

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        """Detects available PureThermal/Lepton cameras."""
        cameras = []
        try:
             result = subprocess.run(
                 ["v4l2-ctl", "--list-devices"], 
                 capture_output=True, text=True, timeout=5
             )
             lines = result.stdout.split('\n')
             for i, line in enumerate(lines):
                 if "PureThermal" in line:
                     # Look for /dev/videoX in subsequent lines
                     for j in range(i+1, min(i+4, len(lines))):
                         if "/dev/video" in lines[j]:
                             dev_path = lines[j].strip()
                             cameras.append({
                                 "name": f"Lepton PureThermal ({dev_path})",
                                 "index": i, # Dummy index
                                 "port": dev_path
                             })
        except Exception as e:
             logger.warning(f"Failed to scan devices: {e}")
        return cameras

    def connect(self, warmup: bool = True):
        """Connects to the Lepton camera."""
        if self.is_connected:
            raise RuntimeError(f"{self} is already connected.")

        # Auto-detect if path not provided
        if self.device_path is None:
            cameras = self.find_cameras()
            if not cameras:
                 raise DeviceNotConnectedError(f"PureThermal/Lepton device not found. Please verify connection.")
            self.device_path = cameras[0]["port"]
        
        logger.info(f"Connecting to Lepton at {self.device_path}")
        
        # Start capture thread
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        
        # Wait for first frame
        start_t = time.perf_counter()
        while time.perf_counter() - start_t < 5.0:
            with self.frame_lock:
                if self.frame_buffer is not None:
                    self.is_connected = True
                    break
            time.sleep(0.1)
            
        if not self.is_connected:
             self.disconnect()
             raise DeviceNotConnectedError(f"Failed to receive frames from {self.device_path}")

        if warmup:
             time.sleep(1.0)

    def _capture_loop(self):
        """Background thread to read Y16 data from v4l2-ctl."""
        # Set format
        try:
            subprocess.run([
                'v4l2-ctl', '-d', self.device_path,
                '--set-fmt-video', f'width={self.width},height={self.height},pixelformat=Y16'
            ], capture_output=True, timeout=5)
        except Exception as e:
            logger.error(f"Failed to set format: {e}")
            return

        # Start streaming
        cmd = [
            'v4l2-ctl', '-d', self.device_path, 
            '--stream-mmap', '--stream-count=0', '--stream-to=-'
        ]
        
        try:
            self.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=self.width * self.height * 2
            )
            
            frame_size = self.width * self.height * 2
            
            while not self.stop_event.is_set():
                data = self.proc.stdout.read(frame_size)
                if len(data) == frame_size:
                    # Convert raw bytes to 16-bit array
                    raw_frame = np.frombuffer(data, dtype=np.uint16).reshape(self.height, self.width)
                    
                    # Post-process
                    processed = self._postprocess_image(raw_frame)
                    
                    with self.frame_lock:
                        self.frame_buffer = processed
                elif len(data) == 0:
                    break
                    
        except Exception as e:
            logger.error(f"Capture loop error: {e}")
        finally:
            if self.proc:
                self.proc.terminate()

    def _postprocess_image(self, raw_frame: NDArray[Any]) -> NDArray[Any]:
        """Convert Y16 raw frame to RGB colormap if configured."""
        if self.colormap:
            # Normalize 16-bit to 8-bit using min-max scaling for best contrast per frame
            # (Adaptive Gain Control style)
            norm = cv2.normalize(raw_frame, None, 0, 255, cv2.NORM_MINMAX)
            gray8 = norm.astype(np.uint8)
            
            # Apply colormap
            # Evaluate the string "cv2.COLORMAP_PLASMA" to get the constant value
            cmap_val = eval(self.colormap) if "cv2." in self.config.colormap else getattr(cv2, self.config.colormap)
            colored = cv2.applyColorMap(gray8, cmap_val)
            
            # OpenCV returns BGR, LeRobot wants RGB
            rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
            return rgb
        else:
            return raw_frame

    def read(self, color_mode: Any = None) -> NDArray[Any]:
        """Read the latest frame."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
            
        with self.frame_lock:
            if self.frame_buffer is None:
                 raise RuntimeError("No frame available")
            return self.frame_buffer.copy()

    def async_read(self, timeout_ms: float = 1000) -> NDArray[Any]:
        """Async read is same as read since we already use a background thread."""
        return self.read()


    def disconnect(self):
        """Stop capture and release resources."""
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        
        if self.proc:
            self.proc.terminate()
            self.proc = None
            
        self.is_connected = False

    def __del__(self):
        self.disconnect()
