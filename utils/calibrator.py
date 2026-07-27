"""
TensorRT INT8 Entropy Calibrator for YOLOv5.

Uses a representative subset of COCO val2017 images to calibrate
activation ranges for INT8 post-training quantization (PTQ).

Algorithm: IInt8EntropyCalibrator2
- Minimizes KL-divergence between FP32 and INT8 activation distributions
- Best general-purpose calibrator for CNN-based vision models
"""

import os
import numpy as np

from utils.common import setup_logger

logger = setup_logger("calibrator")

# TensorRT + PyCUDA imports are deferred to runtime since they
# are only available on NVIDIA GPU environments (Kaggle, not macOS)
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit

    HAS_TENSORRT = True
except ImportError:
    HAS_TENSORRT = False
    logger.warning("TensorRT/PyCUDA not available. Calibrator will not function.")


class YOLOv5EntropyCalibrator:
    """
    Custom INT8 calibrator for YOLOv5 using Entropy Calibration v2.

    This class feeds representative calibration images to TensorRT's
    calibration process, which determines optimal per-layer scaling factors
    to minimize quantization error when converting from FP32 to INT8.

    Usage:
        data_loader = CalibrationDataLoader(images_dir, num_images=500)
        calibrator = YOLOv5EntropyCalibrator(
            data_loader,
            cache_file="outputs/calibration.cache"
        )
        # Pass calibrator to TensorRT builder config
        config.int8_calibrator = calibrator
    """

    def __init__(
        self,
        data_loader,
        cache_file: str = "outputs/calibration.cache",
        input_shape: tuple = (1, 3, 640, 640),
    ):
        if not HAS_TENSORRT:
            raise RuntimeError("TensorRT and PyCUDA are required for INT8 calibration")

        # Initialize parent class
        trt.IInt8EntropyCalibrator2.__init__(self)

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

        Args:
            names: List of input tensor names (unused, required by API)

        Returns:
            List of device memory pointers, or None when data is exhausted
        """
        try:
            batch_data = next(iter(self.data_loader) if self._batch_count == 0 and hasattr(self, '_iter_started') else self._get_next())

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

    def _get_next(self):
        """Get next batch from data loader."""
        if not hasattr(self, '_iterator'):
            self._iterator = iter(self.data_loader)
            self._iter_started = True
        return next(self._iterator)

    def read_calibration_cache(self):
        """
        Read cached calibration data if available.

        TensorRT calls this before starting calibration. If a valid cache
        exists, calibration is skipped entirely (massive time savings).
        """
        if os.path.exists(self.cache_file):
            logger.info(f"Loading calibration cache from: {self.cache_file}")
            with open(self.cache_file, "rb") as f:
                return f.read()
        logger.info("No calibration cache found. Running full calibration...")
        return None

    def write_calibration_cache(self, cache):
        """
        Save calibration data to cache file.

        Called by TensorRT after calibration completes. The cache contains
        per-layer scaling factors and can be reused to skip calibration
        on subsequent engine builds.
        """
        os.makedirs(os.path.dirname(self.cache_file) or ".", exist_ok=True)
        with open(self.cache_file, "wb") as f:
            f.write(cache)
        logger.info(f"Calibration cache saved to: {self.cache_file}")


# We need to make the class inherit from trt.IInt8EntropyCalibrator2 at runtime
# because the import may not be available at module load time (e.g., on macOS)
if HAS_TENSORRT:
    # Dynamically set the base class
    YOLOv5EntropyCalibrator = type(
        "YOLOv5EntropyCalibrator",
        (trt.IInt8EntropyCalibrator2,),
        dict(YOLOv5EntropyCalibrator.__dict__),
    )
