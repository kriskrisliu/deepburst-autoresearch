"""
Sub Task 002 V2: Extract, modify and inject Q/DQ parameters into ONNX model
支持 --exclude_layers 参数，用于过滤 best_scale_dict 中被排除层的参数

This script:
1. Extracts Q/DQ quantizer parameters from INT8 ONNX model
2. Saves original parameters to JSON (for reference)
3. Modifies the scale values using JSON (with optional layer exclusion)
4. Injects modified parameters back into ONNX
5. Builds a new TensorRT engine
"""

import os
import sys
import argparse
import json
import numpy as np
import onnx
from onnx import numpy_helper, helper

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from trt_utils import build_tensorrt_engine


def should_include_tensor(tensor_name, exclude_layers):
    """
    Check if a tensor should be included (not in excluded layers)

    Args:
        tensor_name: Name of the tensor (e.g., "/encoder2/ConvBlock/conv1/input_quantizer/...")
        exclude_layers: List of layer names to exclude (e.g., ["encoder2"])

    Returns:
        bool: True if tensor should be included, False if excluded
    """
    if not exclude_layers:
        return True

    # Check if tensor name contains any excluded layer path
    for excluded_layer in exclude_layers:
        # Match patterns like "/encoder2/" or "/encoder2.ConvBlock/"
        if f"/{excluded_layer}/" in tensor_name:
            return False
        # Also check without leading slash
        if tensor_name.startswith(f"{excluded_layer}/"):
            return False

    return True


def extract_qdq_from_onnx(onnx_path, save_json_path=None):
    """
    Extract Q/DQ node information from ONNX model

    Args:
        onnx_path: Path to ONNX model
        save_json_path: If provided, save original parameters to this JSON file

    Returns:
        model, initializers, constant_values, scale_tensors, zp_tensors, constant_nodes_map
    """
    print(f"Loading ONNX model from: {onnx_path}")
    model = onnx.load(onnx_path)

    # Get initializers (constants like scale and zero_point)
    initializers = {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}
    print(f"Found {len(initializers)} initializers")

    # Build a map of Constant node outputs to their values
    constant_values = {}
    constant_nodes_map = {}  # Map from output name to node
    for node in model.graph.node:
        if node.op_type == "Constant":
            const_name = node.name
            for attr in node.attribute:
                if attr.name == "value":
                    tensor = attr.t
                    arr = numpy_helper.to_array(tensor)
                    for output in node.output:
                        constant_values[output] = arr
                        constant_nodes_map[output] = node

    print(f"Found {len(constant_values)} Constant nodes")

    # Collect Q/DQ nodes
    quantize_nodes = []
    dequantize_nodes = []

    for node in model.graph.node:
        if node.op_type == "QuantizeLinear":
            quantize_nodes.append(node)
        elif node.op_type == "DequantizeLinear":
            dequantize_nodes.append(node)

    print(f"Found {len(quantize_nodes)} QuantizeLinear nodes")
    print(f"Found {len(dequantize_nodes)} DequantizeLinear nodes")

    # Find all scale and zero_point values used by Q/DQ nodes
    # Scale/zero_point can be either in initializers or in Constant nodes
    scale_tensors = {}  # name -> (value, is_initializer)
    zp_tensors = {}     # name -> (value, is_initializer)

    for node in model.graph.node:
        if node.op_type in ["QuantizeLinear", "DequantizeLinear"]:
            if len(node.input) > 1:
                scale_name = node.input[1]
                if scale_name in initializers:
                    scale_tensors[scale_name] = (initializers[scale_name], True)
                elif scale_name in constant_values:
                    scale_tensors[scale_name] = (constant_values[scale_name], False)
            if len(node.input) > 2:
                zp_name = node.input[2]
                if zp_name in initializers:
                    zp_tensors[zp_name] = (initializers[zp_name], True)
                elif zp_name in constant_values:
                    zp_tensors[zp_name] = (constant_values[zp_name], False)

    print(f"Found {len(scale_tensors)} scale tensors")
    print(f"Found {len(zp_tensors)} zero_point tensors")

    # Print original scale statistics
    all_scales = []
    for name, (arr, is_init) in scale_tensors.items():
        all_scales.extend(arr.flatten() if arr.ndim > 0 else [arr])

    if all_scales:
        orig_min = min(all_scales)
        orig_max = max(all_scales)
        orig_mean = sum(all_scales) / len(all_scales)
        print(f"\nOriginal scale statistics:")
        print(f"  Min: {orig_min:.6f}")
        print(f"  Max: {orig_max:.6f}")
        print(f"  Mean: {orig_mean:.6f}")

    # Save to JSON if requested
    if save_json_path:
        save_qdq_to_json(scale_tensors, zp_tensors, save_json_path)

    return model, initializers, constant_values, scale_tensors, zp_tensors, constant_nodes_map


def save_qdq_to_json(scale_tensors, zp_tensors, output_path):
    """
    Save Q/DQ parameters to JSON file

    Args:
        scale_tensors: Dict of scale tensor name -> (value, is_initializer)
        zp_tensors: Dict of zero_point tensor name -> (value, is_initializer)
        output_path: Path to save JSON file
    """
    qdq_data = {
        "description": "Original Q/DQ parameters extracted from INT8 ONNX model",
        "scale_tensors": {},
        "zero_point_tensors": {}
    }

    # Save scales
    for name, (arr, is_init) in scale_tensors.items():
        # Convert numpy array to list for JSON serialization
        if arr.ndim == 0:
            # Scalar
            value = float(arr) if arr.dtype.kind == 'f' else int(arr)
        else:
            # Array
            value = arr.flatten().tolist()

        qdq_data["scale_tensors"][name] = {
            "value": value,
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "is_initializer": is_init
        }

    # Save zero points
    for name, (arr, is_init) in zp_tensors.items():
        if arr.ndim == 0:
            value = float(arr) if arr.dtype.kind == 'f' else int(arr)
        else:
            value = arr.flatten().tolist()

        qdq_data["zero_point_tensors"][name] = {
            "value": value,
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "is_initializer": is_init
        }

    # Calculate statistics
    all_scales = []
    for name, (arr, _) in scale_tensors.items():
        all_scales.extend(arr.flatten() if arr.ndim > 0 else [arr])

    if all_scales:
        qdq_data["statistics"] = {
            "scale_count": len(all_scales),
            "scale_min": float(min(all_scales)),
            "scale_max": float(max(all_scales)),
            "scale_mean": float(sum(all_scales) / len(all_scales))
        }

    # Save to file
    with open(output_path, 'w') as f:
        json.dump(qdq_data, f, indent=2)

    print(f"\nOriginal Q/DQ parameters saved to: {output_path}")
    print(f"  Scale tensors: {len(qdq_data['scale_tensors'])}")
    print(f"  Zero point tensors: {len(qdq_data['zero_point_tensors'])}")


def create_modified_onnx(onnx_path, output_path, scale_multiplier=1.0, scale_divisor=1.0,
                          json_path=None, exclude_layers=None):
    """
    Create a modified ONNX with adjusted quantization parameters

    Args:
        onnx_path: Path to original ONNX model
        output_path: Path to save modified ONNX model
        scale_multiplier: Multiply scales by this factor
        scale_divisor: Divide scales by this factor
        json_path: If provided, load values from this JSON file instead of using multiplier/divisor
        exclude_layers: List of layer names to exclude from JSON value application
    """
    print(f"\n{'='*60}")

    # Load JSON if provided
    json_values = None
    json_format = None  # 'standard' or 'simple'
    excluded_count = 0

    if json_path and os.path.exists(json_path):
        print(f"Loading modification values from JSON: {json_path}")
        with open(json_path, 'r') as f:
            json_values = json.load(f)

        # Detect JSON format
        # Standard format: {"scale_tensors": {...}, "zero_point_tensors": {...}}
        # Simple format: {"tensor_name": value, ...}
        if "scale_tensors" in json_values or "zero_point_tensors" in json_values:
            json_format = "standard"
            print(f"  Detected standard format")
            print(f"  Loaded {len(json_values.get('scale_tensors', {}))} scale modifications")
            print(f"  Loaded {len(json_values.get('zero_point_tensors', {}))} zero_point modifications")
        else:
            json_format = "simple"
            print(f"  Detected simple format")
            print(f"  Loaded {len(json_values)} parameter modifications")

        # Filter out excluded layers if specified
        if exclude_layers:
            print(f"\n  Excluding layers: {exclude_layers}")
            if json_format == "simple":
                # Filter simple format dict
                filtered_values = {}
                for name, value in json_values.items():
                    if should_include_tensor(name, exclude_layers):
                        filtered_values[name] = value
                    else:
                        excluded_count += 1
                        print(f"    Excluded: {name}")
                json_values = filtered_values
            else:
                # Filter standard format
                filtered_scales = {}
                for name, value in json_values.get('scale_tensors', {}).items():
                    if should_include_tensor(name, exclude_layers):
                        filtered_scales[name] = value
                    else:
                        excluded_count += 1
                        print(f"    Excluded: {name}")
                json_values['scale_tensors'] = filtered_scales

            print(f"  Excluded {excluded_count} tensors from excluded layers")
            print(f"  Remaining tensors to modify: {len(json_values if json_format == 'simple' else json_values.get('scale_tensors', {}))}")

    if json_values:
        print(f"Creating modified ONNX using JSON values ({json_format} format)")
    else:
        print(f"Creating modified ONNX with scale_multiplier={scale_multiplier}, scale_divisor={scale_divisor}")
    print(f"{'='*60}")

    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)

    # Get initializers
    initializers = {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}
    print(f"Found {len(initializers)} initializers")

    # Build a map of Constant node outputs to their values
    constant_values = {}
    constant_nodes_map = {}  # Map from output name to node
    for node in model.graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value":
                    tensor = attr.t
                    arr = numpy_helper.to_array(tensor)
                    for output in node.output:
                        constant_values[output] = arr
                        constant_nodes_map[output] = node

    print(f"Found {len(constant_values)} Constant nodes")

    # Find all scale and zero_point values used by Q/DQ nodes
    scale_tensors = {}  # name -> (value, is_initializer)
    zp_tensors = {}     # name -> (value, is_initializer)

    for node in model.graph.node:
        if node.op_type in ["QuantizeLinear", "DequantizeLinear"]:
            if len(node.input) > 1:
                scale_name = node.input[1]
                if scale_name in initializers:
                    scale_tensors[scale_name] = (initializers[scale_name], True)
                elif scale_name in constant_values:
                    scale_tensors[scale_name] = (constant_values[scale_name], False)
            if len(node.input) > 2:
                zp_name = node.input[2]
                if zp_name in initializers:
                    zp_tensors[zp_name] = (initializers[zp_name], True)
                elif zp_name in constant_values:
                    zp_tensors[zp_name] = (constant_values[zp_name], False)

    print(f"Found {len(scale_tensors)} scale tensors")
    print(f"Found {len(zp_tensors)} zero_point tensors")

    # Calculate adjustment factor
    if json_values:
        adjustment_factor = None  # Will use JSON values directly
    else:
        adjustment_factor = scale_multiplier / scale_divisor
        print(f"\nAdjustment factor: {adjustment_factor}")

    # Process scale tensors - build a map of modifications
    # For standard format: json_values.get('scale_tensors', {})
    # For simple format: json_values directly
    if json_values and json_format == "simple":
        json_scale_values = json_values  # Direct dict: {"name": value}
    else:
        json_scale_values = json_values.get('scale_tensors', {}) if json_values else {}

    # Build modifications dict: tensor_name -> new_value
    modifications = {}

    for name, (arr, is_init) in scale_tensors.items():
        # Determine new value
        if json_values and name in json_scale_values:
            # Use value from JSON
            json_val = json_scale_values[name]

            # Handle both standard and simple format
            # Standard: {"value": ..., "shape": ...}
            # Simple: value directly (number)
            if isinstance(json_val, dict) and 'value' in json_val:
                # Standard format
                new_value = json_val['value']
            elif isinstance(json_val, (int, float)):
                # Simple format: direct value
                new_value = float(json_val)
            else:
                # Fallback - keep original
                continue

            modifications[name] = {
                'value': new_value,
                'is_initializer': is_init,
                'shape': list(arr.shape)
            }
            print(f"Will modify scale '{name}': shape={arr.shape}, new_value={new_value}")
        elif json_values:
            # JSON provided but this tensor not in JSON - keep original
            pass
        else:
            # Use multiplier/divisor
            new_arr = arr * adjustment_factor
            modifications[name] = {
                'value': new_arr,
                'is_initializer': is_init,
                'shape': list(arr.shape)
            }
            print(f"Will modify scale '{name}': mean {arr.mean():.6f} -> {new_arr.mean():.6f}")

    # Apply modifications to the model
    modified_count = 0

    # 1. Modify initializers (if any)
    for name, mod_info in modifications.items():
        if mod_info['is_initializer']:
            new_value = mod_info['value']
            if isinstance(new_value, np.ndarray):
                new_arr = new_value
            else:
                # Scalar or list
                shape = mod_info['shape']
                if shape:
                    new_arr = np.full(shape, float(new_value), dtype=np.float32)
                else:
                    new_arr = np.array(new_value, dtype=np.float32)

            # Find and update the initializer
            for i, init in enumerate(model.graph.initializer):
                if init.name == name:
                    model.graph.initializer[i].CopyFrom(numpy_helper.from_array(new_arr.astype(np.float32), name=name))
                    modified_count += 1
                    print(f"Modified initializer '{name}'")
                    break

    # 2. Modify Constant nodes (by updating their value attribute)
    const_output_to_mod = {name: mod_info for name, mod_info in modifications.items() if not mod_info['is_initializer']}

    for node in model.graph.node:
        if node.op_type == "Constant":
            for output_name in node.output:
                if output_name in const_output_to_mod:
                    mod_info = const_output_to_mod[output_name]
                    new_value = mod_info['value']
                    shape = mod_info['shape']

                    # Convert value to array
                    if isinstance(new_value, np.ndarray):
                        new_arr = new_value
                    elif isinstance(new_value, (int, float)):
                        if shape:
                            new_arr = np.full(shape, float(new_value), dtype=np.float32)
                        else:
                            new_arr = np.array(new_value, dtype=np.float32)
                    else:
                        continue

                    # Find and update the value attribute
                    for attr in node.attribute:
                        if attr.name == "value":
                            new_tensor = numpy_helper.from_array(new_arr)
                            attr.t.CopyFrom(new_tensor)
                            modified_count += 1
                            print(f"Modified Constant node '{node.name}' with value {new_value}")
                            break

    print(f"\nTotal tensors modified: {modified_count}")

    # Save modified ONNX
    onnx.save(model, output_path)
    print(f"\nModified ONNX saved to: {output_path}")

    # Verify the modified model
    try:
        onnx.checker.check_model(model)
        print("Modified ONNX model verified successfully")
    except Exception as e:
        print(f"Warning: Modified ONNX model verification failed: {e}")

    modification_type = "json" if json_values else "factor"
    return output_path, modification_type


def build_trt_engine(onnx_path, engine_path, input_shape, max_batch_size, GPU, fp16=True, int8=False):
    """Build TensorRT engine from ONNX model"""
    print(f"\n{'='*60}")
    print(f"Building TensorRT engine from: {onnx_path}")
    print(f"{'='*60}")

    # Set CUDA device - use os.environ to ensure visibility
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU)

    if torch.cuda.is_available():
        torch.cuda.set_device(0)  # Always use device 0 since CUDA_VISIBLE_DEVICES is set
        print(f"Using GPU: {GPU} ({torch.cuda.get_device_name(0)})")

    # Build engine
    build_tensorrt_engine(
        onnx_path=onnx_path,
        engine_path=engine_path,
        input_shape=input_shape,
        max_batch_size=max_batch_size,
        fp16_mode=fp16,
        int8_mode=int8,
        workspace_size=4 << 30,  # 4GB
        device_id=0,  # Always 0 since CUDA_VISIBLE_DEVICES is set
        fp16_layers=[]
    )

    print(f"TensorRT engine saved to: {engine_path}")
    return engine_path


def main():
    parser = argparse.ArgumentParser(description='Sub Task 002 V2: Modify ONNX quantization parameters with layer exclusion support')

    parser.add_argument("--onnx_path", type=str,
                        default='./int8_output/deepburst_int8_zoom1.onnx',
                        help="Path to original INT8 ONNX model")
    parser.add_argument("--output_dir", type=str,
                        default='./int8_output',
                        help="Output directory for modified ONNX and engine")
    parser.add_argument("--denoise_model", type=str, default='zoom1',
                        help="Model name (zoom1 or zoom3) for output file naming")
    parser.add_argument("--save_json", type=str, default=None,
                        help="Save original Q/DQ parameters to this JSON file")
    parser.add_argument("--json_values", type=str, default=None,
                        help="Load modification values from this JSON file (overrides multiplier/divisor)")
    parser.add_argument("--exclude_layers", type=str, default='',
                        help="Comma-separated list of layer names to exclude from JSON value application (e.g., 'encoder2,encoder3')")
    parser.add_argument("--scale_multiplier", type=float, default=1.0,
                        help="Multiply scales by this factor (used if json_values not provided)")
    parser.add_argument("--scale_divisor", type=float, default=1.0,
                        help="Divide scales by this factor (used if json_values not provided)")
    parser.add_argument("--input_shape", type=str, default="(32,1,256,256)",
                        help="Input shape as tuple string")
    parser.add_argument("--max_batch_size", type=int, default=4,
                        help="Maximum batch size")
    parser.add_argument("--GPU", type=str, default='2',
                        help="GPU device index")
    parser.add_argument("--fp16", action="store_true", default=True,
                        help="Enable FP16 precision")
    parser.add_argument("--int8", action="store_true", default=False,
                        help="Enable INT8 quantization")
    parser.add_argument("--skip_build", action="store_true", default=False,
                        help="Skip TensorRT engine build (only modify ONNX)")

    args = parser.parse_args()

    # Parse exclude_layers
    exclude_layers = [layer.strip() for layer in args.exclude_layers.split(',') if layer.strip()]
    if exclude_layers:
        print(f"\nWill exclude these layers from JSON value application: {exclude_layers}")

    # Create output directory if not exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Step 1: Extract and optionally save Q/DQ parameters
    print("\n" + "="*60)
    print("Step 1: Extract original Q/DQ parameters")
    print("="*60)

    # Determine save_json path
    save_json_path = args.save_json
    if save_json_path is None and args.json_values is None:
        # Auto-generate save path if neither save_json nor json_values provided
        save_json_path = os.path.join(args.output_dir, "original_qdq_params.json")

    model, initializers, constant_values, scale_tensors, zp_tensors, constant_nodes_map = \
        extract_qdq_from_onnx(args.onnx_path, save_json_path=save_json_path)

    # Step 2: Create modified ONNX
    if args.json_values:
        # Use base name of JSON file but ensure .onnx extension
        json_basename = os.path.basename(args.json_values)
        # Remove .json extension if present and add .onnx
        if json_basename.endswith('.json'):
            json_basename = json_basename[:-5]

        # Add model name and mixed precision indicator if exclude_layers specified
        if exclude_layers:
            excluded_str = "_".join(exclude_layers)
            output_onnx_path = os.path.join(
                args.output_dir,
                f"deepburst_mixed_{args.denoise_model}_{excluded_str}_modified_from_{json_basename}.onnx"
            )
        else:
            output_onnx_path = os.path.join(
                args.output_dir,
                f"deepburst_int8_{args.denoise_model}_modified_from_{json_basename}.onnx"
            )
    elif args.scale_multiplier != 1.0 or args.scale_divisor != 1.0:
        output_onnx_path = os.path.join(
            args.output_dir,
            f"deepburst_int8_{args.denoise_model}_modified_x{args.scale_multiplier}_div{args.scale_divisor}.onnx"
        )
    else:
        output_onnx_path = os.path.join(args.output_dir, f"deepburst_int8_{args.denoise_model}_modified.onnx")

    print("\n" + "="*60)
    print("Step 2: Create modified ONNX with adjusted scales")
    print("="*60)
    modified_onnx_path, modification_type = create_modified_onnx(
        args.onnx_path,
        output_onnx_path,
        scale_multiplier=args.scale_multiplier,
        scale_divisor=args.scale_divisor,
        json_path=args.json_values,
        exclude_layers=exclude_layers
    )

    # Step 3: Build TensorRT engine
    output_engine_path = modified_onnx_path.replace('.onnx', '.plan')

    if args.skip_build:
        print("\n" + "="*60)
        print("Step 3: Skip TensorRT engine build (--skip_build specified)")
        print("="*60)
        print(f"Modified ONNX saved to: {modified_onnx_path}")
        print(f"To build engine manually, run:")
        print(f"  python build_trt_engine_v2.py --onnx_path {modified_onnx_path} --engine_path {output_engine_path} --fp16 --max_batch_size {args.max_batch_size} --GPU {args.GPU}")
    else:
        print("\n" + "="*60)
        print("Step 3: Build TensorRT engine")
        print("="*60)

        # Parse input shape
        input_shape = eval(args.input_shape)
        if isinstance(input_shape, tuple):
            input_shape = torch.Size(input_shape)

        build_trt_engine(
            onnx_path=modified_onnx_path,
            engine_path=output_engine_path,
            input_shape=input_shape,
            max_batch_size=args.max_batch_size,
            GPU=args.GPU,
            fp16=args.fp16,
            int8=args.int8
        )

    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"Original ONNX: {args.onnx_path}")
    print(f"Modified ONNX: {modified_onnx_path}")
    print(f"TensorRT Engine: {output_engine_path}")
    print(f"Modification type: {modification_type}")
    if args.json_values:
        print(f"JSON values: {args.json_values}")
        if exclude_layers:
            print(f"Excluded layers: {exclude_layers}")
    else:
        print(f"Scale adjustment factor: {args.scale_multiplier / args.scale_divisor}")
    if save_json_path:
        print(f"Saved original params to: {save_json_path}")
    print("="*60)


if __name__ == "__main__":
    main()
