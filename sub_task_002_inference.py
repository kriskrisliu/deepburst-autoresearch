"""
Sub Task 002: TensorRT Inference with modified engine for DeepBurst
This is a modified version of inference_trt.py with explicit engine path.
"""

import os
import sys
import time
import argparse
import numpy as np
import tensorrt as trt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import tifffile as tiff

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_process import test_preprocess_chooseOne, testset, singlebatch_test_save, multibatch_test_save
from utils import save_yaml_test


class TensorRTInference:
    """TensorRT inference wrapper."""

    def __init__(self, engine_path):
        """Initialize TensorRT inference engine."""
        # Load TensorRT engine
        with open(engine_path, "rb") as f:
            self.engine_data = f.read()

        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(self.engine_data)

        if self.engine is None:
            raise RuntimeError("Failed to deserialize TensorRT engine")

        self.context = self.engine.create_execution_context()

        # Helper function to convert trt.Dims to tuple
        def dims_to_tuple(dims):
            return tuple([dims[i] for i in range(len(dims))])

        # Get input/output info (TensorRT 10.x compatible)
        # Use get_tensor_mode to determine input vs output
        self.input_names = []
        self.output_names = []
        for binding in self.engine:
            if self.engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
                self.input_names.append(binding)
            else:
                self.output_names.append(binding)

        print(f"Engine loaded successfully")
        print(f"Input names: {self.input_names}")
        print(f"Output names: {self.output_names}")

        # Allocate buffers using PyTorch tensors (TensorRT 10.x compatible)
        self.inputs = []
        self.outputs = []
        self.bindings = []

        # Determine dtype
        def get_torch_dtype(trt_dtype):
            if trt_dtype == trt.DataType.HALF:
                return torch.float16
            elif trt_dtype == trt.DataType.FLOAT:
                return torch.float32
            elif trt_dtype == trt.DataType.INT32:
                return torch.int32
            else:
                return torch.float32

        for name in self.input_names:
            shape = dims_to_tuple(self.engine.get_tensor_shape(name))
            trt_dtype = self.engine.get_tensor_dtype(name)
            torch_dtype = get_torch_dtype(trt_dtype)
            size = trt.volume(shape)
            # Allocate GPU memory using PyTorch
            device_mem = torch.empty(size, dtype=torch_dtype, device='cuda')
            self.bindings.append(int(device_mem.data_ptr()))
            self.inputs.append({'name': name, 'shape': shape, 'dtype': trt_dtype, 'device_mem': device_mem})

        for name in self.output_names:
            shape = dims_to_tuple(self.engine.get_tensor_shape(name))
            trt_dtype = self.engine.get_tensor_dtype(name)
            torch_dtype = get_torch_dtype(trt_dtype)
            size = trt.volume(shape)
            device_mem = torch.empty(size, dtype=torch_dtype, device='cuda')
            self.bindings.append(int(device_mem.data_ptr()))
            self.outputs.append({'name': name, 'shape': shape, 'dtype': trt_dtype, 'device_mem': device_mem})

    def infer(self, input_tensor):
        """Run inference on input tensor."""
        # Helper function to convert trt.Dims to tuple
        def dims_to_tuple(dims):
            return tuple([dims[i] for i in range(len(dims))])

        # Transfer input to GPU
        input_shape = input_tensor.shape

        # Resize if needed
        if input_tensor.numel() != trt.volume(self.inputs[0]['shape']):
            # Need to resize
            self.context.set_input_shape(self.input_names[0], input_shape)

        # Copy input to GPU buffer
        input_size = input_tensor.numel()
        self.inputs[0]['device_mem'][:input_size].copy_(input_tensor.flatten())

        # Ensure input is copied before inference
        torch.cuda.synchronize()

        # Run inference
        self.context.execute_v2(self.bindings)

        # Synchronize after inference
        torch.cuda.synchronize()

        # Get output
        output_shape = dims_to_tuple(self.engine.get_tensor_shape(self.output_names[0]))
        output_size = trt.volume(output_shape)
        output = self.outputs[0]['device_mem'][:output_size].clone().view(output_shape)

        return output


def test_trt(args):
    """Run TensorRT inference test."""

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)

    # Set CUDA device
    torch.cuda.set_device(0)  # Since CUDA_VISIBLE_DEVICES is set, index 0 is the visible GPU

    # Set paths
    args.patch_x = args.patch_xy
    args.patch_y = args.patch_xy
    args.gap_x = int(args.patch_x * (1 - args.overlap_factor))
    args.gap_y = int(args.patch_y * (1 - args.overlap_factor))

    # Use explicit engine path directly (instead of building from folder)
    engine_path = args.engine_path
    print(f"Loading TensorRT engine from: {engine_path}")

    if not os.path.exists(engine_path):
        raise FileNotFoundError(f"Engine not found at: {engine_path}")

    trt_inference = TensorRTInference(engine_path)

    # Prepare dataset
    im_folder = os.path.join(args.datasets_path, args.datasets_folder)
    img_list = sorted([f for f in os.listdir(im_folder) if f.endswith('.tif')])

    print(f'\n\033[1;31mTesting parameters -----> \033[0m')
    print(args)
    print(f'\033[1;31mStacks to be processed -----> \033[0m')
    print(f'Total stack number -----> {len(img_list)}')
    for img in img_list:
        print(img)

    # Create output directory
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)  # Use makedirs to create parent dirs

    current_time = time.strftime("%Y%m%d%H%M")
    # Replace / with _ in datasets_folder to avoid path issues
    datasets_folder_safe = args.datasets_folder.replace('/', '_')
    output_name = 'DataFolderIs_' + datasets_folder_safe + f'patchXY{args.patch_xy}_TRT_MODIFIED_' + current_time + '_ModelFolderIs_' + args.denoise_model
    output_path1 = os.path.join(args.output_path, output_name)

    if not os.path.exists(output_path1):
        os.makedirs(output_path1, exist_ok=True)

    yaml_name = os.path.join(output_path1, 'para.yaml')
    save_yaml_test(args, yaml_name)

    # Process each image
    for N in range(len(img_list)):
        name_list, noise_img, coordinate_list, test_im_name, img_mean, input_data_type, img_std = \
            test_preprocess_chooseOne(args, N)

        prev_time = time.time()
        denoise_img = torch.zeros((int(noise_img.shape[0] / args.burst), noise_img.shape[1], noise_img.shape[2])).to("cuda", torch.float32)

        test_data = testset(name_list, coordinate_list, noise_img)
        testloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                               num_workers=args.num_workers, pin_memory=True,
                               persistent_workers=True if args.num_workers > 0 else False)

        pbar = tqdm(enumerate(testloader), total=len(testloader), desc="Testing")

        # Warmup
        print('\033[1;33mWarming up model...\033[0m')
        with torch.no_grad():
            sample_noise_patch, _ = test_data[0]
            if args.batch_size > 1:
                dummy_batch = torch.stack([test_data[ii][0] for ii in range(args.batch_size)], dim=0)
            else:
                dummy_batch = sample_noise_patch.unsqueeze(0)
            sample_real = dummy_batch.to('cuda', torch.float16, non_blocking=True)
            sample_real = sample_real.view(-1, *sample_real.shape[-3:])
            # Convert to float32 for TensorRT
            sample_real = sample_real.float()
            for _ in range(3):
                _ = trt_inference.infer(sample_real)
        torch.cuda.synchronize()
        print('\033[1;32mModel warmup completed!\033[0m')

        time_start = time.time()

        with torch.no_grad():
            for iteration, (noise_patch, single_coordinate) in pbar:
                # Convert to float32 for TensorRT
                real = noise_patch.to('cuda', torch.float32, non_blocking=True)
                real = real.view(-1, *real.shape[-3:])

                # Run TensorRT inference
                output = trt_inference.infer(real)

                # Convert output back to float16 for consistency
                output = output.half()

                # Debug: Check output shape
                if iteration == 0:
                    print(f'\n[Debug] First iteration:')
                    print(f'  Input shape: {real.shape}')
                    print(f'  Output shape: {output.shape}')

                # Determine approximate time left
                batches_done = iteration
                batches_left = 1 * len(testloader) - batches_done
                time_left_seconds = int(batches_left * (time.time() - prev_time))
                prev_time = time.time()
                if iteration % 1 == 0:
                    time_end = time.time()
                    time_cost = time_end - time_start
                    pbar.set_postfix(
                        Stack=f"{img_list[N]}",
                        Patch=f"{iteration + 1}/{len(testloader)}",
                        Time_Cost=f"{time_cost:.2f} s",
                        ETA=f"{time_left_seconds:.2f} s"
                    )

                if (iteration + 1) % len(testloader) == 0:
                    print('\n', end=' ')

                # Post-process output
                output_image = output.squeeze(1)

                if (output_image.shape[0] == 1):
                    postprocess_turn = 1
                else:
                    postprocess_turn = output_image.shape[0]

                if (postprocess_turn > 1):
                    for id in range(postprocess_turn):
                        output_patch, stack_start_w, stack_end_w, stack_start_h, stack_end_h, stack_start_s, stack_end_s = \
                            multibatch_test_save(args, single_coordinate, id, output_image)
                        output_patch = output_patch * img_std + img_mean
                        denoise_img[
                            int(stack_start_s / args.burst): int(stack_end_s / args.burst),
                            stack_start_h: stack_end_h,
                            stack_start_w: stack_end_w
                        ] = output_patch
                else:
                    output_patch, stack_start_w, stack_end_w, stack_start_h, stack_end_h, stack_start_s, stack_end_s = \
                        singlebatch_test_save(args, single_coordinate, output_image)
                    output_patch = output_patch * img_std + img_mean
                    denoise_img[int(stack_start_s / args.burst): int(stack_end_s / args.burst),
                               stack_start_h: stack_end_h, stack_start_w: stack_end_w] = output_patch

        torch.cuda.synchronize()
        time_end = time.time()
        time_cost = time_end - time_start

        # Save output
        del noise_img
        output_img = denoise_img.squeeze().float().cpu().numpy() * args.scale_factor
        del denoise_img
        output_img = np.clip(output_img, 0, 65535).astype('int32')

        if input_data_type == 'uint16':
            output_img = np.clip(output_img, 0, 65535)
            output_img = output_img.astype('uint16')
        elif input_data_type == 'int16':
            output_img = np.clip(output_img, -32767, 32767)
            output_img = output_img.astype('int16')
        else:
            output_img = output_img.astype('int32')

        result_file_name = img_list[N].replace('.tif', '') + f'_TRT_MODIFIED_patchXY{args.patch_xy}_output.tif'
        result_name = os.path.join(output_path1, result_file_name)

        if not args.no_save:
            from skimage import io
            io.imsave(result_name, output_img, check_contrast=False)

        print("test result saved in:", result_name)
        output_img_shape = output_img.shape
        output_frames = output_img_shape[0]
        fps = output_frames / time_cost

        print(f"End of Testing! Input image: {img_list[N]}, Output shape: {output_img_shape}, "
              f"Time cost: {time_cost} s, FPS: {fps:.2f}")

        # Calculate SSIM if requested
        if args.ssim:
            mean_ssim_score, std_ssim_score, min_ssim_score, max_ssim_score, scores = \
                calculate_ssim_with_256(output_img, args)
            print(f"Mean SSIM score: {mean_ssim_score:.6f}, Std SSIM score: {std_ssim_score:.6f}, Min SSIM score: {min_ssim_score:.6f}, Max SSIM score: {max_ssim_score:.6f}")
            with open("log_results.txt", "a") as f:
                f.write(f"TensorRT MODIFIED - Args: {args}, Input image: {img_list[N]}, "
                        f"Output shape: {output_img_shape}, Time cost: {time_cost} s, "
                        f"SSIM score: {mean_ssim_score:.6f}\n")

            print(np.sort(scores)[:16])
            print(np.argsort(scores)[:16])


def calculate_ssim_with_256(output_img, args):
    """Calculate SSIM with ground truth."""
    if args.GT_path is None:
        # Extract the last part of the path (e.g., zoom1_P1 from zoom1/zoom1_P1)
        datasets_folder_safe = args.datasets_folder.split('/')[-1]
        # Try different GT path formats
        gt_img_path = None
        # Format 1: zoom1/zoom1_P1 -> zoom1_P1
        img_root = f"GT/DataFolderIs_{datasets_folder_safe}patchXY256_ModelFolderIs_{args.denoise_model}/"
        for root, dirs, files in os.walk(img_root):
            for file in files:
                if file.endswith('.tif'):
                    gt_img_path = os.path.join(root, file)
                    break
            if gt_img_path:
                break
    else:
        gt_img_path = args.GT_path

    if gt_img_path is None or not os.path.exists(gt_img_path):
        print(f"Warning: GT not found at {gt_img_path}")
        return 0.0, 0.0, 0.0, 0.0, []

    gt_img = tiff.imread(gt_img_path)
    from ssim_multi_threads import ssim_volume_parallel
    mean_ssim_score, std_ssim_score, min_ssim_score, max_ssim_score, scores = \
        ssim_volume_parallel(output_img, gt_img)
    return mean_ssim_score, std_ssim_score, min_ssim_score, max_ssim_score, scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sub Task 002: TensorRT Modified Engine Inference for DeepBurst')

    # GPU parameters
    parser.add_argument("--GPU", type=str, default='0', help="GPU device index")

    # Model parameters - Explicit engine path
    parser.add_argument("--engine_path", type=str,
                        default='./int8_output/deepburst_int8_zoom1_modified_x5.0_div1.0.plan',
                        help="Explicit path to TensorRT engine file")
    parser.add_argument("--denoise_model", type=str, default='zoom1',
                        help="Model folder name")

    # Data parameters
    parser.add_argument("--datasets_path", type=str, default='./datasets',
                        help="Path to datasets")
    parser.add_argument("--datasets_folder", type=str, default='zoom1/zoom1_P1',
                        help="Dataset folder")
    parser.add_argument("--patch_xy", type=int, default=256,
                        help="Patch size")
    parser.add_argument("--burst", type=int, default=8,
                        help="Burst size")
    parser.add_argument("--overlap_factor", type=float, default=0,
                        help="Overlap factor")
    parser.add_argument("--scale_factor", type=int, default=1,
                        help="Scale factor")

    # Inference parameters
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers")
    parser.add_argument("--no_save", action='store_true', default=False,
                        help="Don't save output")
    parser.add_argument("--ssim", action='store_true', default=False,
                        help="Calculate SSIM")
    parser.add_argument("--GT_path", type=str, default=None,
                        help="Ground truth path")
    parser.add_argument("--output_path", type=str, default='./results',
                        help="Output path")

    args = parser.parse_args()

    test_trt(args)
