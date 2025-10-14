# Ultralytics YOLO 🚀, AGPL-3.0 license

__version__ = "8.2.64"

import os

# Set ENV Variables (place before imports)
os.environ["OMP_NUM_THREADS"] = "1"  # reduce CPU utilization during training

from ultralytics_custom.data.explorer.explorer import Explorer
from ultralytics_custom.models import NAS, RTDETR, SAM, YOLO, FastSAM, YOLOWorld
from ultralytics_custom.utils import ASSETS, SETTINGS
from ultralytics_custom.utils.checks import check_yolo as checks
from ultralytics_custom.utils.downloads import download

settings = SETTINGS
__all__ = (
    "__version__",
    "ASSETS",
    "YOLO",
    "YOLOWorld",
    "NAS",
    "SAM",
    "FastSAM",
    "RTDETR",
    "checks",
    "download",
    "settings",
    "Explorer",
)
