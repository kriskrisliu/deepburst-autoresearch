import torch
import numpy as np
import tensorrt as trt
import os

class Calibrator(trt.IInt8EntropyCalibrator2):
    """INT8 Calibrator for TensorRT quantization"""
    def __init__(self, calibration_data, input_shape, cache_file=None):
        """
        Args:
            calibration_data: List of calibration input tensors (numpy arrays or torch tensors)
            input_shape: Input shape tuple
            cache_file: Path to save/load calibration cache
        """
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.calibration_data = calibration_data
        self.input_shape = input_shape
        self.cache_file = cache_file
        self.current_index = 0
        
        # Allocate buffer for calibration data
        self.device_input = None
        if len(calibration_data) > 0:
            # Get first sample to determine dtype and allocate buffer
            sample = calibration_data[0]
            if isinstance(sample, torch.Tensor):
                self.device_input = sample.cuda().contiguous()
            else:
                self.device_input = torch.from_numpy(sample).cuda().contiguous()
    
    def get_batch_size(self):
        """Return batch size for calibration"""
        return 1
    
    def get_batch(self, names):
        """Return next batch of calibration data"""
        if self.current_index >= len(self.calibration_data):
            return None
        
        # Get current calibration sample
        sample = self.calibration_data[self.current_index]
        
        # Convert to torch tensor if needed
        if isinstance(sample, torch.Tensor):
            if not sample.is_cuda:
                sample = sample.cuda()
            sample = sample.contiguous()
        else:
            sample = torch.from_numpy(sample).cuda().contiguous()
        
        # Ensure correct shape
        if sample.shape != self.input_shape:
            sample = sample.view(self.input_shape)
        
        # Update device input buffer
        self.device_input = sample
        
        # Return pointer to GPU memory
        self.current_index += 1
        return [int(self.device_input.data_ptr())]
    
    def read_calibration_cache(self):
        """Read calibration cache if exists"""
        if self.cache_file and os.path.exists(self.cache_file):
            print(f'\033[1;33mReading calibration cache from {self.cache_file}\033[0m')
            with open(self.cache_file, 'rb') as f:
                return f.read()
        return None
    
    def write_calibration_cache(self, cache):
        """Write calibration cache to file"""
        if self.cache_file:
            print(f'\033[1;33mWriting calibration cache to {self.cache_file}\033[0m')
            with open(self.cache_file, 'wb') as f:
                f.write(cache)


class TensorRTInference:
    """TensorRT inference engine wrapper using PyTorch CUDA interface (no pycuda required)"""
    def __init__(self, engine_path, input_shape, output_shape, dtype=trt.float16):
        self.engine_path = engine_path
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.dtype = dtype
        self.use_pycuda = False
        
        # Load TensorRT engine
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        
        # Create runtime and engine
        self.runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = self.runtime.deserialize_cuda_engine(engine_data)
        self.context = self.engine.create_execution_context()
        
        # Allocate buffers
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers_pytorch()

    def allocate_buffers_pytorch(self):
        """Allocate GPU memory using PyTorch CUDA interface (modern approach, no pycuda needed)"""
        inputs = []
        outputs = []
        bindings = []
        # Use PyTorch's current CUDA stream
        stream = torch.cuda.current_stream()
        
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            
            # Convert numpy dtype to torch dtype
            if dtype == np.float32:
                torch_dtype = torch.float32
            elif dtype == np.float16:
                torch_dtype = torch.float16
            else:
                torch_dtype = torch.float32
            
            # Allocate host buffer (CPU pinned memory using PyTorch)
            host_mem = torch.empty(size, dtype=torch_dtype, pin_memory=True)
            
            # Allocate device buffer (GPU memory using PyTorch)
            device_mem = torch.empty(size, dtype=torch_dtype, device='cuda')
            
            # Get raw pointer for TensorRT bindings
            bindings.append(device_mem.data_ptr())
            
            if self.engine.binding_is_input(binding):
                inputs.append({'host': host_mem, 'device': device_mem})
            else:
                outputs.append({'host': host_mem, 'device': device_mem})
        
        return inputs, outputs, bindings, stream
    
    def __call__(self, input_tensor):
        """Run inference"""
        if self.use_pycuda:
            return self._infer_pycuda(input_tensor)
        else:
            return self._infer_pytorch(input_tensor)
    
    def _infer_pycuda(self, input_tensor):
        """Run inference using pycuda (deprecated - not used when USE_PYCUDA=False)"""
        raise RuntimeError("PyCUDA inference is not available. Please use PyTorch CUDA interface.")
    
    def _infer_pytorch(self, input_tensor):
        """Run inference using PyTorch CUDA interface (no pycuda needed)"""
        # Ensure input is on GPU and contiguous
        if isinstance(input_tensor, torch.Tensor):
            if not input_tensor.is_cuda:
                input_tensor = input_tensor.cuda()
            input_tensor = input_tensor.contiguous()
        else:
            input_tensor = torch.from_numpy(input_tensor).cuda().contiguous()
        
        # Get actual input shape
        actual_input_shape = input_tensor.shape  # e.g., (32, 1, 256, 256) for batch_size=4
        input_size = input_tensor.numel()
        
        # Get expected input shape from TensorRT
        # For this model, input_shape is (burst, 1, H, W), e.g., (8, 1, 256, 256)
        expected_input_shape = self.input_shape  # This is (burst, 1, H, W)
        expected_input_size = np.prod(expected_input_shape)
        
        # Check if input is batched (larger than expected single input)
        if input_size > expected_input_size:
            # Input is batched: (batch_size * burst, 1, H, W)
            batch_size = input_size // expected_input_size
            if input_size % expected_input_size != 0:
                raise RuntimeError(f"Input size {input_size} is not a multiple of expected size {expected_input_size}. "
                                 f"Input shape: {actual_input_shape}, Expected: {expected_input_shape}")
            
            # Process each sample in the batch separately
            # Reshape to (batch_size, burst, 1, H, W)
            batched_input = input_tensor.view(batch_size, *expected_input_shape)
            
            # Process each sample and collect outputs
            # IMPORTANT: Process sequentially and ensure each output is correctly extracted
            outputs = []
            for i in range(batch_size):
                single_input = batched_input[i]  # (burst, 1, H, W)
                single_output = self._infer_single(single_input, sync=True)  # Force sync for each sample
                outputs.append(single_output)
            
            # Concatenate outputs along batch dimension
            # Each output is (1, 1, H, W), so concatenated result is (batch_size, 1, H, W)
            output_tensor = torch.cat(outputs, dim=0)
            return output_tensor
        else:
            # Single input, process directly
            return self._infer_single(input_tensor, sync=False)
    
    def _infer_single(self, input_tensor, sync=True):
        """Run inference for a single input (burst, 1, H, W)
        
        Args:
            input_tensor: Input tensor of shape (burst, 1, H, W)
            sync: If True, force synchronization after inference (needed for batch processing)
        """
        # Flatten input for copying
        input_flat = input_tensor.view(-1)
        input_size = input_flat.numel()
        
        # Get buffer size
        buffer_size = self.inputs[0]['device'].numel()
        
        # Check if input fits in buffer
        if input_size > buffer_size:
            raise RuntimeError(f"Input size {input_size} exceeds buffer size {buffer_size}")
        
        # Clear input buffer to avoid contamination from previous inference
        self.inputs[0]['device'][:input_size].zero_()
        
        # Copy input to TensorRT input buffer
        self.inputs[0]['device'][:input_size].copy_(input_flat)
        
        # Ensure input is copied before inference
        torch.cuda.synchronize()
        
        # Get CUDA stream handle for TensorRT
        try:
            stream_handle = self.stream.cuda_stream
        except AttributeError:
            try:
                stream_handle = torch.cuda.current_stream().cuda_stream
            except:
                stream_handle = None
        
        # Clear output buffer to avoid contamination
        output_buffer_size = self.outputs[0]['device'].numel()
        self.outputs[0]['device'].zero_()
        
        # Run inference - always use synchronous execution for accuracy
        # This ensures output is ready before we read it
        self.context.execute_v2(bindings=self.bindings)
        
        # Force synchronization to ensure output is ready
        if sync:
            torch.cuda.synchronize()
        
        # Get output from TensorRT output buffer
        # For single input (burst, 1, H, W), model output is (1, 1, H, W)
        # The output_shape passed to __init__ is (batch_size, 1, H, W) for max batch
        # For single input, output is (1, 1, H, W)
        H, W = self.output_shape[2], self.output_shape[3]
        single_output_shape = (1, 1, H, W)
        output_size = np.prod(single_output_shape)
        
        # Extract output - only take the first output (for single sample)
        # The output buffer might be larger (for batch), but we only need the first sample
        # Use clone() to ensure we get a copy that won't be overwritten
        output_tensor = self.outputs[0]['device'][:output_size].clone().view(single_output_shape).contiguous()
        
        # Ensure correct dtype
        if self.dtype == trt.float16:
            output_tensor = output_tensor.half()
        else:
            output_tensor = output_tensor.float()
        
        return output_tensor  # Shape: (1, 1, H, W)


def build_tensorrt_engine(onnx_path, engine_path, input_shape, max_batch_size=1, fp16_mode=True, int8_mode=False, workspace_size=1<<30, calibration_data=None, device_id=0, fp16_layers=None):
    """Build TensorRT engine from ONNX model

    Args:
        onnx_path: Path to ONNX model
        engine_path: Path to save TensorRT engine
        input_shape: Input shape tuple
        max_batch_size: Maximum batch size
        fp16_mode: Use FP16 precision
        int8_mode: Use INT8 quantization (input/output layers remain FP16)
        workspace_size: Workspace size in bytes
        calibration_data: List of calibration samples for INT8 quantization (optional)
        device_id: CUDA device ID to use
        fp16_layers: List of layer name patterns to force FP16 precision (e.g., ['encoder1', 'decoder6'])
    """
    if fp16_layers is None:
        fp16_layers = []
    import torch
    # Set CUDA device
    if torch.cuda.is_available():
        torch.cuda.set_device(device_id)
        print(f"Building TensorRT engine on GPU {device_id}: {torch.cuda.get_device_name(device_id)}")

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    
    # Parse ONNX model
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            print('\033[1;31mFailed to parse ONNX model. Errors:\033[0m')
            for error in range(parser.num_errors):
                error_msg = parser.get_error(error)
                print(f'\033[1;31m  Error {error}: {error_msg}\033[0m')
            raise RuntimeError("Failed to parse ONNX model")
    
    print('\033[1;32mONNX model parsed successfully\033[0m')

    # Set specific layers to FP16 precision if requested
    if fp16_layers and len(fp16_layers) > 0:
        print(f'\033[1;33mSetting FP16 for layers: {fp16_layers}\033[0m')
        fp16_layer_count = 0
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            layer_name = layer.name
            # Check if layer matches any of the fp16 layer patterns
            for pattern in fp16_layers:
                if pattern in layer_name:
                    try:
                        layer.precision = trt.float16
                        fp16_layer_count += 1
                    except Exception as e:
                        print(f'\033[1;33mWarning: Could not set FP16 for layer {layer_name}: {e}\033[0m')
                    break
        print(f'\033[1;32mSet {fp16_layer_count} layers to FP16\033[0m')

    # Print network information for debugging
    num_inputs = network.num_inputs
    num_outputs = network.num_outputs
    print(f'\033[1;33mNetwork has {num_inputs} input(s) and {num_outputs} output(s)\033[0m')
    for i in range(num_inputs):
        input_tensor = network.get_input(i)
        print(f'\033[1;33m  Input {i}: {input_tensor.name}, shape: {input_tensor.shape}\033[0m')
    
    # Configure builder
    config = builder.create_builder_config()
    # Use new API if available (TensorRT 8.5+)
    if hasattr(config, 'set_memory_pool_limit'):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size)
    else:
        # Fallback to deprecated API
        config.max_workspace_size = workspace_size
    
    # Configure precision modes
    if int8_mode:
        # INT8 quantization with FP16 input/output layers
        if builder.platform_has_fast_int8:
            # Enable both INT8 and FP16 for mixed precision
            config.set_flag(trt.BuilderFlag.INT8)
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
            print('\033[1;32mTensorRT INT8 quantization enabled\033[0m')
            
            # Set input and output tensors to FP16
            # This ensures input/output layers remain in FP16 while internal layers use INT8
            try:
                for i in range(network.num_inputs):
                    input_tensor = network.get_input(i)
                    # Set input tensor dtype to FP16
                    input_tensor.dtype = trt.float16
                    print(f'\033[1;33mInput tensor {input_tensor.name} set to FP16\033[0m')
                
                for i in range(network.num_outputs):
                    output_tensor = network.get_output(i)
                    # Set output tensor dtype to FP16
                    output_tensor.dtype = trt.float16
                    print(f'\033[1;33mOutput tensor {output_tensor.name} set to FP16\033[0m')
                
                print('\033[1;32mConfiguration: Input/Output layers = FP16, Internal layers = INT8\033[0m')
            except Exception as e:
                print(f'\033[1;33mWarning: Could not explicitly set input/output to FP16: {e}\033[0m')
                print('\033[1;33mTensorRT will use automatic mixed precision (INT8 + FP16)\033[0m')
            
            # Set up INT8 calibrator
            if calibration_data is None or len(calibration_data) == 0:
                print('\033[1;31mError: INT8 quantization requires calibration data!\033[0m')
                print('\033[1;33mFalling back to FP16 mode\033[0m')
                config.clear_flag(trt.BuilderFlag.INT8)  # Remove INT8 flag
                int8_mode = False
                if builder.platform_has_fast_fp16:
                    config.set_flag(trt.BuilderFlag.FP16)
                    print('\033[1;32mTensorRT FP16 mode enabled instead\033[0m')
            else:
                # Create calibration cache file path
                cache_file = engine_path.replace('.engine', '_calib.cache')
                calibrator = Calibrator(calibration_data, input_shape, cache_file)
                config.int8_calibrator = calibrator
                print(f'\033[1;33mUsing {len(calibration_data)} samples for INT8 calibration\033[0m')
        else:
            print('\033[1;33mWarning: INT8 requested but not supported on this platform, falling back to FP16\033[0m')
            int8_mode = False
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                print('\033[1;32mTensorRT FP16 mode enabled instead\033[0m')
    elif fp16_mode:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print('\033[1;32mTensorRT FP16 mode enabled\033[0m')
        else:
            print('\033[1;33mWarning: FP16 requested but not supported on this platform, using FP32\033[0m')
    else:
        print('\033[1;32mTensorRT FP32 mode enabled\033[0m')
    
    # Set input shape
    # Note: For this model, input_shape is (burst, channels, H, W) = (8, 1, 256, 256)
    # This is NOT a batch dimension - it's the burst dimension which is fixed
    profile = builder.create_optimization_profile()
    input_name = network.get_input(0).name

    # Get the actual input shape from ONNX model
    input_shape_onnx = network.get_input(0).shape
    print(f'\033[1;33mONNX model input shape from network: {input_shape_onnx}\033[0m')
    print(f'\033[1;33mProvided input_shape: {input_shape}\033[0m')

    # Convert input_shape to tuple if it's not already
    if isinstance(input_shape, torch.Size):
        input_shape = tuple(input_shape)

    # Check if ONNX has static dimensions that differ from requested
    # If ONNX has static (8,1,256,256) but we want (32,1,256,256),
    # we need to use ONNX's shape since it's static
    if input_shape_onnx and all(s is not None and s > 0 for s in input_shape_onnx):
        # ONNX has static shape, use it
        actual_input_shape = tuple(input_shape_onnx)
        print(f'\033[1;33mUsing ONNX static shape: {actual_input_shape}\033[0m')
    else:
        # Use provided shape
        actual_input_shape = input_shape

    print(f'\033[1;33mUsing input shape for TensorRT: {actual_input_shape}\033[0m')

    # Set optimization profile: min, opt, max shapes
    # For static ONNX, use the ONNX shape for all three
    try:
        profile.set_shape(input_name, actual_input_shape, actual_input_shape, actual_input_shape)
        config.add_optimization_profile(profile)
        print(f'\033[1;32mTensorRT optimization profile set successfully\033[0m')
    except Exception as e:
        print(f'\033[1;31mError setting optimization profile: {e}\033[0m')
        raise RuntimeError(f"Failed to set optimization profile: {e}")
    
    # Build engine
    print('\033[1;33mBuilding TensorRT engine (this may take a while)...\033[0m')
    try:
        # Use new API if available (TensorRT 8.5+)
        if hasattr(builder, 'build_serialized_network'):
            serialized_engine = builder.build_serialized_network(network, config)
            if serialized_engine is None:
                engine = None
            else:
                runtime = trt.Runtime(logger)
                engine = runtime.deserialize_cuda_engine(serialized_engine)
        else:
            # Fallback to deprecated API
            engine = builder.build_engine(network, config)
    except Exception as e:
        print(f'\033[1;31mError building TensorRT engine: {e}\033[0m')
        print('\033[1;33mThis might be due to:\033[0m')
        print('  1. ONNX model has incompatible operations for TensorRT')
        print('  2. Input shape mismatch between ONNX model and TensorRT configuration')
        print('  3. Model contains dynamic operations that TensorRT cannot optimize')
        print('\033[1;33mTry:\033[0m')
        print('  - Using --tensorrt_fp32 instead of FP16')
        print('  - Checking if the ONNX model is valid')
        print('  - Using torch.compile() or torch.jit instead of TensorRT')
        raise RuntimeError(f"Failed to build TensorRT engine: {e}")
    
    if engine is None:
        print('\033[1;31mTensorRT engine build returned None\033[0m')
        print('\033[1;33mThis usually means:\033[0m')
        print('  - The ONNX model contains operations TensorRT cannot handle')
        print('  - Input/output shape configuration is incorrect')
        print('  - There are errors in the network structure')
        raise RuntimeError("Failed to build TensorRT engine (returned None)")
    
    # Save engine
    with open(engine_path, 'wb') as f:
        f.write(engine.serialize())
    
    print(f'\033[1;32mTensorRT engine saved to: {engine_path}\033[0m')
    return engine


def convert_to_onnx(model, dummy_input, onnx_path, input_names=['input'], output_names=['output']):
    """Convert PyTorch model to ONNX"""
    print('\033[1;33mConverting PyTorch model to ONNX...\033[0m')
    print(f'\033[1;33mInput shape: {dummy_input.shape}\033[0m')
    model.eval()
    
    # Try with higher opset version for better compatibility
    # Use opset 13 or higher for better support of dynamic shapes and control flow
    try:
        opset_version = 13
    except:
        opset_version = 11
    
    with torch.no_grad():
        try:
            # First, try to trace the model to check if it can be traced
            traced_model = torch.jit.trace(model, dummy_input)
            print('\033[1;32mModel can be traced successfully\033[0m')
        except Exception as e:
            print(f'\033[1;33mWarning: Model tracing failed: {e}\033[0m')
            print('\033[1;33mTrying ONNX export directly...\033[0m')
        
        # Export to ONNX
        # For this model, input shape is fixed (burst, 1, H, W), so we don't need dynamic axes
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            # Don't use dynamic axes for this model - input shape is fixed
            dynamic_axes=None,
            verbose=False
        )
    
    # Verify ONNX model
    try:
        import onnx
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print(f'\033[1;32mONNX model saved and verified: {onnx_path}\033[0m')
        # Print input/output shapes
        if len(onnx_model.graph.input) > 0:
            input_shape = [dim.dim_value if dim.dim_value > 0 else '?' for dim in onnx_model.graph.input[0].type.tensor_type.shape.dim]
            print(f'\033[1;33mONNX input shape: {input_shape}\033[0m')
        if len(onnx_model.graph.output) > 0:
            output_shape = [dim.dim_value if dim.dim_value > 0 else '?' for dim in onnx_model.graph.output[0].type.tensor_type.shape.dim]
            print(f'\033[1;33mONNX output shape: {output_shape}\033[0m')
    except Exception as e:
        print(f'\033[1;33mWarning: ONNX model verification failed: {e}\033[0m')
        print(f'\033[1;32mONNX model saved (unverified): {onnx_path}\033[0m')
    
    return onnx_path
