"""
Build TensorRT FP16 Engine from ONNX model using trt_utils
"""

import os
import sys
import argparse

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from trt_utils import build_tensorrt_engine


def main():
    parser = argparse.ArgumentParser(description='Build TensorRT FP16/INT8 Engine from ONNX model')

    parser.add_argument("--onnx_path", type=str, required=True,
                        help="Path to ONNX model")
    parser.add_argument("--engine_path", type=str, required=True,
                        help="Path to save TensorRT engine")
    parser.add_argument("--input_shape", type=str, default="(8,1,256,256)",
                        help="Input shape as tuple string, e.g., '(8,1,256,256)'")
    parser.add_argument("--max_batch_size", type=int, default=4,
                        help="Maximum batch size")
    parser.add_argument("--fp16", action="store_true", default=True,
                        help="Enable FP16 precision")
    parser.add_argument("--int8", action="store_true", default=False,
                        help="Enable INT8 quantization (requires calibration)")
    parser.add_argument("--workspace_size", type=int, default=4,
                        help="Workspace size in GB")
    parser.add_argument("--GPU", type=str, default='0',
                        help="GPU device index")
    parser.add_argument("--fp16_layers", type=str, default='',
                        help="Comma-separated list of layer name patterns to keep as FP16 (e.g., 'encoder1,encoder5,decoder6')")

    args = parser.parse_args()

    # Parse fp16_layers
    fp16_layers = []
    if args.fp16_layers:
        fp16_layers = [l.strip() for l in args.fp16_layers.split(',') if l.strip()]
        print(f"FP16 layers: {fp16_layers}")

    # Load calibration data if INT8 mode
    calibration_data = None
    if args.int8:
        calib_path = args.onnx_path.replace('.onnx', '_calib.pt')
        if os.path.exists('./int8_output/calibration_data.pt'):
            calib_path = './int8_output/calibration_data.pt'
        if os.path.exists(calib_path):
            print(f"Loading calibration data from: {calib_path}")
            calibration_data = torch.load(calib_path)
            print(f"Loaded {len(calibration_data)} calibration samples")

    # Set CUDA device before building
    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.GPU))
        print(f"Using GPU: {args.GPU} ({torch.cuda.get_device_name(0)})")

    # Parse input shape
    input_shape = eval(args.input_shape)
    if isinstance(input_shape, tuple):
        input_shape = torch.Size(input_shape)

    print(f"Input shape: {input_shape}")

    # Build engine
    build_tensorrt_engine(
        onnx_path=args.onnx_path,
        engine_path=args.engine_path,
        input_shape=input_shape,
        max_batch_size=args.max_batch_size,
        fp16_mode=args.fp16,
        int8_mode=args.int8,
        workspace_size=args.workspace_size << 30,  # Convert GB to bytes
        device_id=int(args.GPU),
        fp16_layers=fp16_layers,
        calibration_data=calibration_data
    )

    print("\n" + "="*60)
    print("TensorRT engine build complete!")
    print("="*60)


if __name__ == "__main__":
    main()
