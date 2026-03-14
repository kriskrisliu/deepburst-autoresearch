"""
DeepBurst Mixed Precision Quantization Script
排除特定层的INT8量化，保持FP16精度
Using NVIDIA Model Optimizer (modelopt) for quantization and ONNX export
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_DeepBurst_v7 import DeepBurst
import modelopt.torch.quantization as mtq
from modelopt.torch._deploy.utils import OnnxBytes, get_onnx_bytes_and_metadata


def load_model(model_path, device='cuda'):
    """Load the DeepBurst model from checkpoint."""
    model = DeepBurst(
        in_channels=1,
        out_channels=1,
        f_maps=64,
        burst_size=8,
    )

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    # Convert to half precision for quantization
    model = model.half()
    model.eval()

    return model


def disable_quantizers_for_module(module, module_name=""):
    """
    Recursively disable all quantizers in a module.
    """
    # Common quantizer attribute names in modelopt
    quantizer_attrs = ['_input_quantizer', '_weight_quantizer', '_output_quantizer',
                       'input_quantizer', 'weight_quantizer', 'output_quantizer']

    disabled_count = 0
    for attr_name in quantizer_attrs:
        if hasattr(module, attr_name):
            quantizer = getattr(module, attr_name)
            if quantizer is not None and hasattr(quantizer, 'enable'):
                quantizer.enable = False
                disabled_count += 1
            elif quantizer is not None and hasattr(quantizer, '_enabled'):
                quantizer._enabled = False
                disabled_count += 1

    # Recursively process submodules
    for child_name, child_module in module.named_children():
        disabled_count += disable_quantizers_for_module(child_module, f"{module_name}.{child_name}")

    return disabled_count


def quantize_model_mixed_precision(model, calibration_data_loader=None, exclude_layers=None):
    """
    Quantize model to INT8 using modelopt with mixed precision.

    Args:
        model: The model to quantize
        calibration_data_loader: Calibration data
        exclude_layers: List of layer names to exclude from quantization (keep FP16)
    """
    exclude_layers = exclude_layers or []

    # Use INT8 quantization config - use the predefined config
    config = mtq.INT8_DEFAULT_CFG

    # Print info about excluded layers
    if exclude_layers:
        print(f"\nExcluding layers from INT8 quantization: {exclude_layers}")

    # Perform quantization first
    if calibration_data_loader is not None:
        def forward_loop(model):
            for batch in calibration_data_loader:
                model(batch)

        quantized_model = mtq.quantize(model, config, forward_loop=forward_loop)
    else:
        quantized_model = mtq.quantize(model, config)

    # Disable quantization for excluded layers by manually disabling their quantizers
    if exclude_layers:
        print("\nApplying layer exclusion (setting to FP16)...")
        for layer_name in exclude_layers:
            try:
                # Find the module by name
                if hasattr(quantized_model, layer_name):
                    target_module = getattr(quantized_model, layer_name)
                    disabled_count = disable_quantizers_for_module(target_module, layer_name)
                    if disabled_count > 0:
                        print(f"  Disabled quantization for: {layer_name} ({disabled_count} quantizers)")
                    else:
                        print(f"  Warning: No quantizers found in {layer_name}")
                else:
                    print(f"  Warning: Layer {layer_name} not found in model")
            except Exception as e:
                print(f"  Warning: Could not disable quantization for {layer_name}: {e}")

    return quantized_model


def export_to_onnx(model, input_shape, onnx_save_path, device, weights_dtype="fp16"):
    """Export the quantized model to ONNX format."""
    # Create input tensor
    input_tensor = torch.randn(input_shape, dtype=torch.float16).to(device)
    model_name = os.path.basename(onnx_save_path).replace(".onnx", "")

    onnx_bytes, _ = get_onnx_bytes_and_metadata(
        model=model,
        dummy_input=(input_tensor,),
        weights_dtype=weights_dtype,
        model_name=model_name,
    )
    onnx_bytes_obj = OnnxBytes.from_bytes(onnx_bytes)

    # Write the onnx model to the specified directory
    os.makedirs(os.path.dirname(onnx_save_path), exist_ok=True)
    onnx_bytes_obj.write_to_disk(os.path.dirname(onnx_save_path), clean_dir=False)

    print(f"ONNX model saved to: {onnx_save_path}")


def create_calibration_data_from_dataset(args, num_samples=32):
    """Create calibration data from the dataset."""
    from data_process import test_preprocess_chooseOne
    import tifffile as tiff

    calibration_data = []

    # Load dataset - datasets_path/datasets_folder contains the images
    # e.g., datasets_path = ./datasets, datasets_folder = zoom1/zoom1_P1
    im_folder = os.path.join(args.datasets_path, args.datasets_folder)
    img_list = sorted([f for f in os.listdir(im_folder) if f.endswith('.tif')])

    print(f"Creating calibration data from {len(img_list)} images in {im_folder}...")

    # Load at least one image for calibration
    if len(img_list) > 0:
        img_path = os.path.join(im_folder, img_list[0])
        img = tiff.imread(img_path)
        print(f"Loaded calibration image shape: {img.shape}")

        # Get args for preprocessing
        args.patch_x = args.patch_xy
        args.patch_y = args.patch_xy
        args.gap_x = int(args.patch_x * (1 - args.overlap_factor))
        args.gap_y = int(args.patch_y * (1 - args.overlap_factor))

        # Preprocess
        name_list, noise_img, coordinate_list, test_im_name, img_mean, input_data_type, img_std = \
            test_preprocess_chooseOne(args, 0)

        # Extract patches
        from data_process import testset
        test_data = testset(name_list, coordinate_list, noise_img)

        # Get calibration samples
        num_samples = min(num_samples, len(test_data))
        for i in range(num_samples):
            patch, _ = test_data[i]
            # Convert to float16 and move to GPU
            patch = patch.to('cuda', torch.float16)
            # Reshape to [batch * burst, 1, H, W]
            patch = patch.view(-1, *patch.shape[-3:])
            calibration_data.append(patch)

        print(f"Created {len(calibration_data)} calibration samples")

    return calibration_data


def print_quant_summary_by_layer(model):
    """Print quantization summary grouped by layer."""
    print("\n" + "="*60)
    print("Quantization Summary by Layer")
    print("="*60)

    # Try to get quantizer status
    for name, module in model.named_modules():
        # Check if module has quantizer attributes
        has_input_quant = hasattr(module, '_input_quantizer') and module._input_quantizer is not None
        has_weight_quant = hasattr(module, '_weight_quantizer') and module._weight_quantizer is not None
        has_output_quant = hasattr(module, '_output_quantizer') and module._output_quantizer is not None

        if has_input_quant or has_weight_quant or has_output_quant:
            status = []
            if has_input_quant:
                status.append("input")
            if has_weight_quant:
                status.append("weight")
            if has_output_quant:
                status.append("output")
            print(f"  {name}: {', '.join(status)}")

    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Quantize DeepBurst to INT8 with Mixed Precision')

    # Model parameters
    parser.add_argument("--pth_path", type=str, default='./models',
                        help="Path to model checkpoints")
    parser.add_argument("--denoise_model", type=str, default='zoom1',
                        help="Model folder name")
    parser.add_argument("--output_dir", type=str, default='./int8_output',
                        help="Output directory for ONNX model")

    # Data parameters for calibration
    parser.add_argument("--datasets_path", type=str, default='./datasets',
                        help="Path to datasets")
    parser.add_argument("--datasets_folder", type=str, default='zoom1_P1',
                        help="Dataset folder for calibration")
    parser.add_argument("--patch_xy", type=int, default=256,
                        help="Patch size")
    parser.add_argument("--burst", type=int, default=8,
                        help="Burst size")
    parser.add_argument("--overlap_factor", type=float, default=0,
                        help="Overlap factor")

    # Quantization parameters
    parser.add_argument("--calib_samples", type=int, default=32,
                        help="Number of calibration samples")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for calibration and export")
    parser.add_argument("--scale_factor", type=int, default=1,
                        help="Scale factor for image intensity")

    # Mixed precision parameters
    parser.add_argument("--exclude_layers", type=str, default='',
                        help="Comma-separated list of layer names to exclude from INT8 quantization (e.g., 'encoder2,fusion')")

    args = parser.parse_args()

    # Parse exclude_layers
    exclude_layers = [layer.strip() for layer in args.exclude_layers.split(',') if layer.strip()]

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Find model checkpoint
    model_path = os.path.join(args.pth_path, args.denoise_model)
    model_files = [f for f in os.listdir(model_path) if f.endswith('.pth')]
    model_files.sort()

    if not model_files:
        raise FileNotFoundError(f"No .pth files found in {model_path}")

    model_file = model_files[-1]  # Use the latest model
    full_model_path = os.path.join(model_path, model_file)
    print(f"Loading model from: {full_model_path}")

    # Load model
    model = load_model(full_model_path, device)
    print("Model loaded successfully")

    # Print model summary
    print("\nModel Summary:")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Print available layers for exclusion
    if exclude_layers:
        print("\nAvailable encoder/decoder layers in model:")
        for name, _ in model.named_modules():
            if any(keyword in name for keyword in ['encoder', 'decoder', 'fusion', 'final_conv']):
                if not '.' in name or name.count('.') <= 1:  # Top-level modules only
                    print(f"  - {name}")

    # Create calibration data
    print("\nCreating calibration data...")
    calibration_data = create_calibration_data_from_dataset(args, args.calib_samples)

    # Quantize model with mixed precision
    print("\nQuantizing model to INT8 with mixed precision...")
    if exclude_layers:
        print(f"Layers excluded from quantization: {exclude_layers}")
    quantized_model = quantize_model_mixed_precision(model, calibration_data, exclude_layers)

    # Print quantization summary
    print("\nQuantization Summary:")
    mtq.print_quant_summary(quantized_model)

    # Print layer-wise summary
    print_quant_summary_by_layer(quantized_model)

    # Prepare input shape for ONNX export
    # Input: (batch_size * burst_size, 1, H, W)
    input_shape = (args.batch_size * args.burst, 1, args.patch_xy, args.patch_xy)

    # Export to ONNX
    if exclude_layers:
        layer_suffix = '_'.join(exclude_layers)
        onnx_filename = f"deepburst_mixed_{args.denoise_model}_{layer_suffix}.onnx"
    else:
        onnx_filename = f"deepburst_int8_{args.denoise_model}.onnx"

    onnx_path = os.path.join(args.output_dir, onnx_filename)
    print(f"\nExporting to ONNX with shape: {input_shape}")
    export_to_onnx(quantized_model, input_shape, onnx_path, device, weights_dtype="fp16")

    print("\n" + "="*60)
    print("Mixed precision quantization complete!")
    print(f"ONNX model saved to: {onnx_path}")
    if exclude_layers:
        print(f"Excluded layers (FP16): {exclude_layers}")
    print("="*60)


if __name__ == "__main__":
    main()
