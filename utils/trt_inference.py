"""
TensorRT inference wrapper.

Provides a clean interface to load serialized TensorRT engines
and run inference with proper memory management.
"""

import os
import numpy as np

from utils.common import setup_logger

logger = setup_logger("trt_inference")

# Deferred imports for GPU-only dependencies
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit

    HAS_TENSORRT = True
except ImportError:
    HAS_TENSORRT = False
    logger.warning("TensorRT/PyCUDA not available.")


class TRTInference:
    """
    TensorRT engine inference wrapper.

    Handles:
    - Engine deserialization from file
    - Input/output buffer allocation (host + device)
    - Synchronous inference with data transfer
    - Proper resource cleanup

    Usage:
        engine = TRTInference("outputs/yolov5n_int8.engine")
        output = engine.infer(input_tensor)  # input: (1,3,640,640) float32
        engine.cleanup()
    """

    def __init__(self, engine_path: str):
        if not HAS_TENSORRT:
            raise RuntimeError("TensorRT and PyCUDA are required for inference")

        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"Engine file not found: {engine_path}")

        self.engine_path = engine_path
        self.logger = trt.Logger(trt.Logger.WARNING)

        # Deserialize engine
        logger.info(f"Loading TensorRT engine: {engine_path}")
        runtime = trt.Runtime(self.logger)
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize engine: {engine_path}")

        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        # Allocate buffers
        self._allocate_buffers()

        logger.info(
            f"Engine loaded: {self.num_inputs} input(s), {self.num_outputs} output(s)"
        )

    def _allocate_buffers(self):
        """Allocate host and device memory for all engine bindings."""
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.output_shapes = []
        self.num_inputs = 0
        self.num_outputs = 0

        # TRT 10.x API: num_io_tensors / get_tensor_name / get_tensor_mode
        # TRT 8.x API: num_bindings / binding_is_input
        use_modern_api = hasattr(self.engine, 'num_io_tensors')

        if use_modern_api:
            num_tensors = self.engine.num_io_tensors
            for i in range(num_tensors):
                name = self.engine.get_tensor_name(i)
                shape = self.engine.get_tensor_shape(name)
                dtype = trt.nptype(self.engine.get_tensor_dtype(name))
                is_input = (self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT)
                self._alloc_binding(name, shape, dtype, is_input)
        else:
            # Legacy TRT 8.x API
            for i in range(self.engine.num_bindings):
                name = self.engine.get_binding_name(i)
                shape = self.engine.get_binding_shape(i)
                dtype = trt.nptype(self.engine.get_binding_dtype(i))
                is_input = self.engine.binding_is_input(i)
                self._alloc_binding(name, shape, dtype, is_input)

    def _alloc_binding(self, name, shape, dtype, is_input):
        """Allocate memory for a single binding."""
        size = int(np.prod(shape))
        nbytes = size * np.dtype(dtype).itemsize

        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(nbytes)

        binding = {
            "name": name,
            "shape": tuple(shape),
            "dtype": dtype,
            "host": host_mem,
            "device": device_mem,
            "nbytes": nbytes,
        }

        if is_input:
            self.inputs.append(binding)
            self.num_inputs += 1
        else:
            self.outputs.append(binding)
            self.output_shapes.append(tuple(shape))
            self.num_outputs += 1

        self.bindings.append(int(device_mem))

        logger.info(
            f"  {'Input' if is_input else 'Output'} "
            f"'{name}': shape={tuple(shape)}, dtype={np.dtype(dtype).name}"
        )

    def infer(self, input_data: np.ndarray) -> list:
        """
        Run inference on input data.

        Args:
            input_data: Input tensor, shape matching engine input (e.g., 1,3,640,640)

        Returns:
            List of output numpy arrays
        """
        # Copy input to host buffer
        input_flat = input_data.astype(self.inputs[0]["dtype"]).ravel()
        np.copyto(self.inputs[0]["host"], input_flat)

        # Transfer input to device
        cuda.memcpy_htod_async(
            self.inputs[0]["device"],
            self.inputs[0]["host"],
            self.stream,
        )

        # Execute inference — try modern API first, fall back to legacy
        if hasattr(self.context, 'set_tensor_address'):
            # TRT 10.x API
            for inp in self.inputs:
                self.context.set_tensor_address(inp["name"], int(inp["device"]))
            for out in self.outputs:
                self.context.set_tensor_address(out["name"], int(out["device"]))
            self.context.execute_async_v3(stream_handle=self.stream.handle)
        elif hasattr(self.context, 'execute_async_v2'):
            # TRT 8.x API
            self.context.execute_async_v2(
                bindings=self.bindings,
                stream_handle=self.stream.handle,
            )
        else:
            raise RuntimeError("No compatible TensorRT execution API found")

        # Transfer outputs back to host
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out["host"], out["device"], self.stream)

        # Synchronize
        self.stream.synchronize()

        # Collect outputs
        results = []
        for i, out in enumerate(self.outputs):
            result = out["host"].reshape(self.output_shapes[i]).copy()
            results.append(result)

        return results

    def get_input_shape(self) -> tuple:
        """Get the expected input shape."""
        return self.inputs[0]["shape"]

    def get_output_shapes(self) -> list:
        """Get output shapes."""
        return self.output_shapes

    def cleanup(self):
        """Free GPU resources."""
        # Device memory is freed automatically by PyCUDA's garbage collector
        # but we can help by clearing references
        del self.context
        del self.engine
        logger.info("TRT engine resources cleaned up")

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass


class CUDATimer:
    """
    High-precision GPU timer using CUDA events.
    More accurate than Python time.time() for GPU operations.
    """

    def __init__(self):
        if not HAS_TENSORRT:
            raise RuntimeError("PyCUDA required for CUDA timing")
        self.start_event = cuda.Event()
        self.end_event = cuda.Event()

    def start(self, stream=None):
        """Record start event."""
        self.start_event.record(stream)

    def stop(self, stream=None):
        """Record end event and synchronize."""
        self.end_event.record(stream)
        self.end_event.synchronize()

    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        return self.start_event.time_till(self.end_event)
