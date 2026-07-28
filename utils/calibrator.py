"""
TensorRT INT8 Entropy Calibrator for YOLOv5.

Uses a representative subset of COCO val2017 images to calibrate
activation ranges for INT8 post-training quantization (PTQ).

Supports TensorRT 8.x and 10.x API differences.
"""

import os
import numpy as np

from utils.common import setup_logger

logger = setup_logger("calibrator")

# TensorRT + PyCUDA imports are deferred to runtime since they
# are only available on NVIDIA GPU environments (Kaggle, not macOS)
HAS_TENSORRT = False
HAS_CALIBRATOR_API = False
_CalibratorBase = object  # default base class

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    HAS_TENSORRT = True

    # Try to find the calibrator base class — location varies by TRT version
    # TRT 8.x: trt.IInt8EntropyCalibrator2
    # TRT 10.x: may be in tensorrt module or tensorrt.IInt8EntropyCalibrator2
    _CalibratorBase = None
    for attr_name in ["IInt8EntropyCalibrator2", "IInt8Calibrator"]:
        if hasattr(trt, attr_name):
            _CalibratorBase = getattr(trt, attr_name)
            HAS_CALIBRATOR_API = True
            logger.info(f"Found calibrator base class: trt.{attr_name}")
            break

    if _CalibratorBase is None:
        # TRT 10.x lean package may not include calibrator classes
        _CalibratorBase = object
        logger.warning(
            f"TensorRT {trt.__version__} does not expose calibrator classes. "
            "INT8 calibration via Python API is not available. "
            "Will use ultralytics export for INT8 instead."
        )

except ImportError:
    logger.warning("TensorRT/PyCUDA not available. Calibrator will not function.")


class YOLOv5EntropyCalibrator(_CalibratorBase):
    """
    Custom INT8 calibrator for YOLOv5 using Entropy Calibration v2.

    This class feeds representative calibration images to TensorRT's
    calibration process, which determines optimal per-layer scaling factors
    to minimize quantization error when converting from FP32 to INT8.

    NOTE: If TensorRT's calibrator API is not available (TRT 10.x lean),
    this class cannot be used. Use ultralytics export for INT8 instead.
    """

    def __init__(
        self,
        data_loader,
        cache_file: str = "outputs/calibration.cache",
        input_shape: tuple = (1, 3, 640, 640),
    ):
        if not HAS_TENSORRT:
            raise RuntimeError("TensorRT and PyCUDA are required for INT8 calibration")
        if not HAS_CALIBRATOR_API:
            raise RuntimeError(
                f"TensorRT {trt.__version__} does not expose calibrator classes. "
                "Use ultralytics export for INT8 instead."
            )

        # Initialize parent class
        super().__init__()

        self.data_loader = data_loader
        self.cache_file = cache_file
        self.batch_size = input_shape[0]
        self.input_shape = input_shape

        # Calculate buffer size
        self.nbytes = int(np.prod(input_shape) * np.dtype(np.float32).itemsize)

        # Allocate GPU memory for calibration input
        self.d_input = cuda.mem_alloc(self.nbytes)

        self._batch_count = 0
        self._total_batches = len(data_loader)
        self._iterator = None

        logger.info(
            f"Calibrator initialized: {self._total_batches} batches, "
            f"input_shape={input_shape}, cache={cache_file}"
        )

    def get_batch_size(self) -> int:
        """Return the batch size used for calibration."""
        return self.batch_size

    def get_batch(self, names):
        """
        Fetch the next batch of calibration data.

        Called by TensorRT during the calibration process. Loads the next
        batch of preprocessed images and transfers them to GPU memory.
        """
        try:
            if self._iterator is None:
                self._iterator = iter(self.data_loader)

            batch_data = next(self._iterator)

            # Ensure contiguous float32 array
            batch_data = np.ascontiguousarray(batch_data, dtype=np.float32)

            # Transfer to GPU
            cuda.memcpy_htod(self.d_input, batch_data)

            self._batch_count += 1
            if self._batch_count % 50 == 0:
                logger.info(f"Calibration progress: {self._batch_count}/{self._total_batches} batches")

            return [int(self.d_input)]

        except StopIteration:
            logger.info(f"Calibration complete: processed {self._batch_count} batches")
            return None

    def read_calibration_cache(self):
        """Read cached calibration data if available."""
        if os.path.exists(self.cache_file):
            logger.info(f"Loading calibration cache from: {self.cache_file}")
            with open(self.cache_file, "rb") as f:
                return f.read()
        logger.info("No calibration cache found. Running full calibration...")
        return None

    def write_calibration_cache(self, cache):
        """Save calibration data to cache file."""
        os.makedirs(os.path.dirname(self.cache_file) or ".", exist_ok=True)
        with open(self.cache_file, "wb") as f:
            f.write(cache)
        logger.info(f"Calibration cache saved to: {self.cache_file}")
