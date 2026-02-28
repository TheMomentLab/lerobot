import dataclasses
from dataclasses import dataclass

from ..configs import CameraConfig


@CameraConfig.register_subclass("lepton")
@dataclass
class LeptonCameraConfig(CameraConfig):
    """Configuration class for FLIR Lepton thermal cameras (via PureThermal).

    Attributes:
        fps: Requested frames per second. Lepton is limited to ~9Hz. Defaults to 9.
        width: Requested frame width. Lepton 3.5 is 160. Defaults to 160.
        height: Requested frame height. Lepton 3.5 is 120. Defaults to 120.
        device_path: Specific path to video device (e.g. /dev/video0). If None, auto-detects PureThermal.
        colormap: OpenCV colormap to apply (e.g. "cv2.COLORMAP_PLASMA"). Defaults to "cv2.COLORMAP_PLASMA".
                  Can be set to None to return raw 1-channel images (but scaled to 8-bit).
    """

    fps: int = 9
    width: int = 160
    height: int = 120
    device_path: str | None = None
    colormap: str | None = "cv2.COLORMAP_PLASMA"

    def __post_init__(self) -> None:
        # Basic validation
        if self.fps > 9:
            raise ValueError(f"Lepton cameras typically max out at 9 FPS. You requested {self.fps}.")
