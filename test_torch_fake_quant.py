import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import time
import datetime
from skimage import io
import numpy as np
from model_DeepBurst_v7 import DeepBurst

from utils import save_yaml_test
from data_process import test_preprocess_chooseOne, testset, singlebatch_test_save, multibatch_test_save
from tqdm import tqdm
# from skimage.metrics import structural_similarity as calculate_ssim
import tifffile as tiff
from ssim_multi_threads import ssim_volume_parallel
import torch.cuda.nvtx as nvtx
from types import MethodType
import json


def calculate_ssim_with_256(output_img, args):
    if args.GT_path is None:
        # Extract the last part of the path (e.g., zoom1_P1 from zoom1/zoom1_P1)
        datasets_folder_safe = args.datasets_folder.split('/')[-1]
        img_root = f"GT/DataFolderIs_{datasets_folder_safe}patchXY256_ModelFolderIs_{args.denoise_model}/"
        # find the .tif file in the folder and subfolders, assert there is only one
        gt_img_path = None
        for root, dirs, files in os.walk(img_root):
            for file in files:
                if file.endswith('.tif'):
                    gt_img_path = os.path.join(root, file)
                    break
    else:
        gt_img_path = args.GT_path
    gt_img = tiff.imread(gt_img_path)
    assert gt_img_path is not None, "No .tif file found in the folder and subfolders"
    assert os.path.exists(gt_img_path), "GT image path does not exist"
    assert output_img.dtype == gt_img.dtype, "Output and GT image have different data types"
    mean_ssim_score, std_ssim_score, min_ssim_score, max_ssim_score, scores = ssim_volume_parallel(output_img, gt_img)
    return mean_ssim_score, std_ssim_score, min_ssim_score, max_ssim_score, scores

def test(args):

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)

    args.patch_x = args.patch_xy
    args.patch_y = args.patch_xy
    args.gap_x = int(args.patch_x * (1 - args.overlap_factor))
    args.gap_y = int(args.patch_y * (1 - args.overlap_factor))
    args.ngpu = args.GPU.count(',') + 1
    # Allow larger batch_size for better GPU utilization
    if not hasattr(args, 'batch_size') or args.batch_size is None:
        args.batch_size = args.ngpu

    # Enable cuDNN benchmark for optimal performance (only if input shapes are fixed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = True
        print('\033[1;32mcuDNN benchmark enabled for faster inference\033[0m')

    print('\033[1;31mTesting parameters -----> \033[0m')
    print(args)

    model_path = os.path.join(args.pth_path, args.denoise_model)
    model_file_list = list(os.walk(model_path, topdown=False))[-1][-1]
    model_list = [item for item in model_file_list if '.pth' in item]
    model_list.sort()
    try:
        args.model_list = model_list[-1]
    except Exception as e:
        print('\033[1;31mThere is no .pth file in the models directory! \033[0m')
        sys.exit()
    args.model_list_length = 1

    for i in range(len(model_list)):
        aaa = model_list[i]
        if '.yaml' in aaa:
            yaml_name = model_list[i]
            del model_list[i]
    print('If there are multiple models, only the last one will be used for denoising.')
    model_list.sort()
    model_list[:-1] = []

    im_folder = os.path.join(args.datasets_path, args.datasets_folder)
    img_list = list(os.walk(im_folder, topdown=False))[-1][-1]
    img_list.sort()
    print('\033[1;31mStacks to be processed -----> \033[0m')
    print('Total stack umber -----> ', len(img_list))
    for img in img_list: print(img)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    current_time = datetime.datetime.now().strftime("%Y%m%d%H%M")
    # Replace / with _ in datasets_folder to avoid path issues
    datasets_folder_safe = args.datasets_folder.replace('/', '_')
    output_name = 'DataFolderIs_' + datasets_folder_safe + f'patchXY{args.patch_xy}_' + current_time + '_ModelFolderIs_' + args.denoise_model
    output_path1 = os.path.join(args.output_path, output_name)

    if not os.path.exists(output_path1):
        os.mkdir(output_path1)

    yaml_name = os.path.join(output_path1, 'para.yaml')

    save_yaml_test(args, yaml_name)

    denoise_generator = DeepBurst(
        in_channels=1,
        out_channels=1,
        f_maps=64,
        burst_size=args.burst,
    )

    if torch.cuda.is_available():
        print('\033[1;31mUsing {} GPU(s) for testing -----> \033[0m'.format(torch.cuda.device_count()))
        denoise_generator = denoise_generator.cuda()
        # denoise_generator = nn.DataParallel(denoise_generator, device_ids=range(args.ngpu))
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    pth_count = 0

    if '.pth' in args.model_list:
        pth_count = pth_count + 1
        pth_name = args.model_list
        output_path = os.path.join(output_path1, pth_name.replace('.pth', ''))

        if not os.path.exists(output_path):
                os.mkdir(output_path)

        model_name = os.path.join(args.pth_path, args.denoise_model, pth_name)
        if isinstance(denoise_generator, nn.DataParallel):
            denoise_generator.module.load_state_dict(torch.load(model_name, weights_only=True))  # parallel
            denoise_generator.module = denoise_generator.module.half()

        else:
            denoise_generator.load_state_dict(torch.load(model_name))  # not parallel
            denoise_generator = denoise_generator.half()

        if args.fake_quant:
            def quant_dequant_per_tensor_sym(x, bits=8, scale=None):
                if scale is None:
                    if x.min().abs() > x.max():
                        scale = x.min().abs() / 2**(bits-1)
                    else:
                        scale = x.max() / (2**(bits-1) - 1)
                qx = torch.round(x / scale).clamp(-2**(bits-1), 2**(bits-1) - 1)
                return qx * scale

            MSE_OPTIMIZE = (True, False)[0]
            ACT_QUANT = (True, False)[0]
            best_scale_dict = {}
            def quant_dequant_conv2d_forward(self, x):
                if not ACT_QUANT:
                    return self._conv_forward(x, self.weight, self.bias)
                if not hasattr(self, "scale"):
                    bits = 8
                    if x.min().abs() > x.max():
                        self.scale = x.min().abs() / 2**(bits-1)
                    else:
                        self.scale = x.max() / (2**(bits-1) - 1)
                    
                    if MSE_OPTIMIZE:
                        z_fp = self._conv_forward(x, self.weight, self.bias)
                        best_mse = float('inf')
                        init_mse = None
                        best_scale = None
                        pbar = tqdm(range(100, 30, -1), desc=f"MSE - {self.module_name}")
                        for rr in pbar:
                            ratio = rr / 100
                            scale = x.abs().max() * ratio / (2**(bits-1) - 1)
                            xq = quant_dequant_per_tensor_sym(x, bits=8, scale=scale)
                            zq = self._conv_forward(xq, self.weight, self.bias)
                            mse = ((z_fp - zq) ** 2).mean()
                            if init_mse is None:
                                init_mse = mse
                            if mse < best_mse:
                                best_mse = mse
                                best_scale = scale
                            pbar.set_postfix(
                                {
                                    "01_InitMSE": f"{init_mse:.3e}",
                                    "02_BestMSE": f"{best_mse:.3e}",
                                    "03_BestScale": f"{best_scale:.3e}",
                                }
                            )
                        self.scale = best_scale
                        # Ensure JSON-serializable (convert possible 0-dim tensor to Python float)
                        if isinstance(best_scale, torch.Tensor):
                            best_scale_dict["/"+self.module_name.replace(".", "/") + "/input_quantizer/Constant_1_output_0"] = best_scale.item()
                        else:
                            best_scale_dict["/"+self.module_name.replace(".", "/") + "/input_quantizer/Constant_1_output_0"] = float(best_scale)
                x = quant_dequant_per_tensor_sym(x, bits=8, scale=self.scale)
                return self._conv_forward(x, self.weight, self.bias)

            with torch.no_grad():
                for name, module in denoise_generator.named_modules():
                    setattr(module, "module_name", name)
                    if isinstance(module, nn.Conv2d):
                        if any([nn in name for nn in args.exclude_layers]):
                            continue
                        module.weight.data.copy_(quant_dequant_per_tensor_sym(module.weight.data.clone()))
                        module.forward = MethodType(quant_dequant_conv2d_forward, module)
                        print(f"Quantized {name}")
                    elif isinstance(module, nn.Conv3d):
                        raise NotImplementedError("Conv3d is not supported for fake quantization")
                    elif isinstance(module, nn.Linear):
                        raise NotImplementedError("Linear is not supported for fake quantization")
                    elif isinstance(module, nn.BatchNorm2d):
                        raise NotImplementedError("BatchNorm2d is not supported for fake quantization")

        denoise_generator.eval()
        denoise_generator.cuda()

        # Use torch.compile() for PyTorch 2.0+ (better than JIT for most cases)
        if args.use_compile:
            print('\033[1;31mCompiling model with torch.compile() for acceleration...\033[0m')
            try:
                # Use 'reduce-overhead' mode for inference, with dynamic shapes support
                # This helps with batch_size > 1 scenarios
                denoise_generator = torch.compile(denoise_generator, mode='reduce-overhead', dynamic=True)
                print('\033[1;32mModel successfully compiled with torch.compile()!\033[0m')
            except Exception as e:
                print(f'\033[1;33mWarning: torch.compile() failed: {e}. Continuing without compilation.\033[0m')

        # Convert to JIT if requested (fallback option)
        elif args.use_jit:
            print('\033[1;31mConverting model to TorchScript JIT for acceleration...\033[0m')
            try:
                # Script the model
                with torch.no_grad():
                    scripted_model = torch.jit.script(denoise_generator)
                    scripted_model = scripted_model.eval()
                denoise_generator = scripted_model
                print('\033[1;32mModel successfully converted to JIT!\033[0m')
            except Exception as e:
                print(f'\033[1;33mWarning: JIT scripting failed: {e}. Continuing without JIT.\033[0m')

        args.print_img_name = True
        print('Testing the last model by default:')
        for N in range(len(img_list)):
            name_list, noise_img, coordinate_list,test_im_name, img_mean, input_data_type, img_std = test_preprocess_chooseOne(args, N)
            # print(len(name_list))
            prev_time = time.time()
            denoise_img = torch.zeros((int(noise_img.shape[0] / args.burst), noise_img.shape[1], noise_img.shape[2])).to("cuda", torch.float16)
            input_img = np.zeros((int(noise_img.shape[0] / args.burst), noise_img.shape[1], noise_img.shape[2]))

            test_data = testset(name_list, coordinate_list, noise_img)
            testloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                   num_workers=args.num_workers, pin_memory=True, persistent_workers=True if args.num_workers > 0 else False)
            pbar = tqdm(enumerate(testloader), total=len(testloader), desc="Testing")

            # Model warmup for optimal performance
            # Skip warmup if using torch.compile() with batch_size > 1 to avoid shape tracking issues
            # The first few iterations will serve as warmup naturally
            # if not (args.use_compile and args.batch_size > 1):
            if True:
                print('\033[1;33mWarming up model...\033[0m')
                with torch.no_grad():
                    try:
                        # Use dummy data for warmup instead of consuming the first batch from testloader
                        # This prevents the first batch from being skipped in actual inference
                        # Get shape directly from test_data without consuming testloader
                        sample_noise_patch, _ = test_data[0]
                        # Create a dummy batch with the same shape for warmup
                        # If batch_size > 1, repeat the sample to match batch_size
                        if args.batch_size > 1:
                            dummy_batch = torch.stack([test_data[ii][0] for ii in range(args.batch_size)], dim=0)
                        else:
                            dummy_batch = sample_noise_patch.unsqueeze(0)
                        sample_real = dummy_batch.to('cuda', torch.float16, non_blocking=True)
                        sample_real = sample_real.view(-1, *sample_real.shape[-3:])
                        # Run warmup iterations
                        for _ in range(3):
                            _ = denoise_generator(sample_real)
                        torch.cuda.synchronize()
                        print('\033[1;32mModel warmup completed!\033[0m')
                    except (IndexError, Exception) as e:
                        if args.use_compile:
                            print(f'\033[1;33mWarning: Warmup skipped (will warmup during first iterations): {e}\033[0m')
                        else:
                            print(f'\033[1;33mWarning: No data for warmup, skipping...\033[0m')
            else:
                print('\033[1;33mSkipping explicit warmup (torch.compile with batch_size > 1). First iterations will warmup naturally.\033[0m')

            time_start = time.time()
            # Performance profiling
            total_data_load_time = 0.0
            total_transfer_to_gpu_time = 0.0
            total_inference_time = 0.0
            total_transfer_to_cpu_time = 0.0
            total_postprocess_time = 0.0
            total_iterations = 0

            # Lists to store CUDA events for each iteration
            transfer_to_gpu_starters = []
            transfer_to_gpu_enders = []
            inference_starters = []
            inference_enders = []
            transfer_to_cpu_starters = []
            transfer_to_cpu_enders = []

            nvtx.range_push("infer_core")

            with torch.no_grad():
                for iteration, (noise_patch, single_coordinate) in pbar:
                    # Measure data loading time (already done by DataLoader, but we can measure iteration overhead)

                    # Pre-trained models are loaded into memory and the sub-stacks are directly fed into the model.
                    # Use CUDA Event for GPU transfer timing
                    transfer_to_gpu_starter = torch.cuda.Event(enable_timing=True)
                    transfer_to_gpu_ender = torch.cuda.Event(enable_timing=True)
                    transfer_to_gpu_starters.append(transfer_to_gpu_starter)
                    transfer_to_gpu_enders.append(transfer_to_gpu_ender)

                    transfer_to_gpu_starter.record()
                    real = noise_patch.to('cuda', torch.float16, non_blocking=True)
                    real = real.view(-1, *real.shape[-3:])
                    transfer_to_gpu_ender.record()

                    # Inference timing using CUDA Event
                    inference_starter = torch.cuda.Event(enable_timing=True)
                    inference_ender = torch.cuda.Event(enable_timing=True)
                    inference_starters.append(inference_starter)
                    inference_enders.append(inference_ender)

                    inference_starter.record()
                    # Use PyTorch model
                    output = denoise_generator(real)
                    inference_ender.record()

                    # Debug: Check output shape consistency
                    if iteration == 0:
                        print(f'\n[Debug] First iteration:')
                        print(f'  Input shape: {real.shape}')
                        print(f'  Output shape: {output.shape}')
                        if args.batch_size and args.batch_size > 1:
                            print(f'  Expected output shape: [{args.batch_size}, 1, H, W]')
                        else:
                            print(f'  Expected output shape: [1, 1, H, W]')
                        # Check if output values are reasonable
                        print(f'  Output min: {output.min().item():.6f}, max: {output.max().item():.6f}, mean: {output.mean().item():.6f}')
                    # output = output.unsqueeze(0)

                    # real = real.reshape(*real.shape[:2], real.shape[2] // args.burst, args.burst, *real.shape[3:]).mean(dim=3)

                    # Determine approximate time left
                    batches_done = iteration
                    batches_left = 1 * len(testloader) - batches_done
                    time_left_seconds = int(batches_left * (time.time() - prev_time))
                    prev_time = time.time()
                    if iteration % 1 == 0:
                        time_end = time.time()
                        time_cost = time_end - time_start  # datetime.timedelta(seconds= (time_end - time_start))
                        pbar.set_postfix(
                            Model=f"{pth_name}",
                            Stack=f"{img_list[N]}",
                            Patch=f"{iteration + 1}/{len(testloader)}",
                            Time_Cost=f"{time_cost:.2f} s",
                            ETA=f"{time_left_seconds:.2f} s"
                        )

                    if (iteration + 1) % len(testloader) == 0:
                        print('\n', end=' ')

                    # Enhanced sub-stacks are sequentially output from the network
                    # Use CUDA Event for GPU->CPU transfer timing
                    transfer_to_cpu_starter = torch.cuda.Event(enable_timing=True)
                    transfer_to_cpu_ender = torch.cuda.Event(enable_timing=True)
                    transfer_to_cpu_starters.append(transfer_to_cpu_starter)
                    transfer_to_cpu_enders.append(transfer_to_cpu_ender)

                    transfer_to_cpu_starter.record()
                    # output_image = np.squeeze(output.cpu().detach().numpy(), axis=(1))
                    output_image = output.squeeze(1)
                    transfer_to_cpu_ender.record()

                    postprocess_start = time.time()
                    if (output_image.shape[0] == 1):
                        postprocess_turn = 1
                    else:
                        postprocess_turn = output_image.shape[0]

                    # The final enhanced stack can be obtained by stitching all sub-stacks.
                    if (postprocess_turn > 1):
                        for id in range(postprocess_turn):
                            output_patch, stack_start_w, stack_end_w, stack_start_h, stack_end_h, stack_start_s, stack_end_s = multibatch_test_save(
                                args, single_coordinate, id, output_image)
                            output_patch = output_patch * img_std + img_mean
                            # raw_patch = raw_patch * img_std + img_mean
                            denoise_img[
                                int(stack_start_s / args.burst): int(stack_end_s / args.burst),
                                stack_start_h: stack_end_h,
                                stack_start_w: stack_end_w
                            ] = output_patch
                            # input_img[int(stack_start_s / args.burst): int(stack_end_s / args.burst), stack_start_h: stack_end_h,
                            # stack_start_w: stack_end_w] \
                            #     = raw_patch
                    else:
                        output_patch, stack_start_w, stack_end_w, stack_start_h, stack_end_h, stack_start_s, stack_end_s = singlebatch_test_save(
                            args, single_coordinate, output_image)
                        output_patch=output_patch * img_std + img_mean
                        # raw_patch=raw_patch * img_std + img_mean
                        denoise_img[int(stack_start_s / args.burst): int(stack_end_s / args.burst), stack_start_h: stack_end_h, stack_start_w: stack_end_w] \
                            = output_patch
                        # input_img[int(stack_start_s / args.burst): int(stack_end_s / args.burst), stack_start_h: stack_end_h, stack_start_w: stack_end_w] \
                        #     = raw_patch
                    postprocess_time = time.time() - postprocess_start
                    total_postprocess_time += postprocess_time

                    total_iterations += 1
                    # Data load time will be calculated after synchronizing GPU events

                # Synchronize all CUDA events and calculate timings (unified synchronization outside iteration)
                torch.cuda.synchronize()

                # Calculate final wall-clock time
                time_end = time.time()
                time_cost = time_end - time_start

                # Calculate GPU operation timings from events
                for i in range(len(transfer_to_gpu_starters)):
                    transfer_to_gpu_time = transfer_to_gpu_starters[i].elapsed_time(transfer_to_gpu_enders[i]) / 1000.0  # Convert ms to seconds
                    total_transfer_to_gpu_time += transfer_to_gpu_time

                    inference_time = inference_starters[i].elapsed_time(inference_enders[i]) / 1000.0  # Convert ms to seconds
                    total_inference_time += inference_time

                    transfer_to_cpu_time = transfer_to_cpu_starters[i].elapsed_time(transfer_to_cpu_enders[i]) / 1000.0  # Convert ms to seconds
                    total_transfer_to_cpu_time += transfer_to_cpu_time

                # Calculate data loading time (overhead not measured by GPU events)
                # This is approximate - the difference between total iteration time and GPU operations
                total_data_load_time = time_cost - total_transfer_to_gpu_time - total_inference_time - total_transfer_to_cpu_time - total_postprocess_time
                if total_data_load_time < 0:
                    total_data_load_time = 0.0  # Avoid negative values due to timing inaccuracies
                del noise_img
                output_img = denoise_img.squeeze().float().cpu().numpy() * args.scale_factor
                del denoise_img
                output_img=np.clip(output_img, 0, 65535).astype('int32')
                # Save inference image
                if input_data_type == 'uint16':
                    output_img=np.clip(output_img, 0, 65535)
                    output_img = output_img.astype('uint16')

                elif input_data_type == 'int16':
                    output_img=np.clip(output_img, -32767, 32767)
                    output_img = output_img.astype('int16')

                else:
                    output_img = output_img.astype('int32')

                result_file_name = img_list[N].replace('.tif', '') + '_' + pth_name.replace('.pth','') + f'patchXY{args.patch_xy}_output.tif'
                result_name = os.path.join(output_path, result_file_name)
                if not args.no_save:
                    io.imsave(result_name, output_img, check_contrast=False)
                print("test result saved in:", result_name)
                output_img_shape = output_img.shape
                output_frames = output_img_shape[0]
                fps = output_frames / time_cost

                # Print performance breakdown
                print('\n' + '='*60)
                print('\033[1;31mPerformance Breakdown:\033[0m')
                print('='*60)
                if total_iterations > 0:
                    avg_data_load = total_data_load_time / total_iterations * 1000
                    avg_transfer_gpu = total_transfer_to_gpu_time / total_iterations * 1000
                    avg_inference = total_inference_time / total_iterations * 1000
                    avg_transfer_cpu = total_transfer_to_cpu_time / total_iterations * 1000
                    avg_postprocess = total_postprocess_time / total_iterations * 1000

                    total_measured = (total_data_load_time + total_transfer_to_gpu_time +
                                    total_inference_time + total_transfer_to_cpu_time +
                                    total_postprocess_time)

                    print(f'Total iterations: {total_iterations}')
                    print(f'Data loading overhead:     {total_data_load_time:.3f}s ({total_data_load_time/total_measured*100:.1f}%) - avg {avg_data_load:.2f}ms/iter')
                    print(f'CPU->GPU transfer:         {total_transfer_to_gpu_time:.3f}s ({total_transfer_to_gpu_time/total_measured*100:.1f}%) - avg {avg_transfer_gpu:.2f}ms/iter')
                    print(f'Model inference:           {total_inference_time:.3f}s ({total_inference_time/total_measured*100:.1f}%) - avg {avg_inference:.2f}ms/iter')
                    print(f'GPU->CPU transfer:         {total_transfer_to_cpu_time:.3f}s ({total_transfer_to_cpu_time/total_measured*100:.1f}%) - avg {avg_transfer_cpu:.2f}ms/iter')
                    print(f'Post-processing:           {total_postprocess_time:.3f}s ({total_postprocess_time/total_measured*100:.1f}%) - avg {avg_postprocess:.2f}ms/iter')
                    print(f'Total measured time:       {total_measured:.3f}s')
                    print(f'Total wall-clock time:     {time_cost:.3f}s')
                    print('='*60)
                    print('\033[1;33mBottleneck Analysis:\033[0m')
                    times = {
                        'Data loading': total_data_load_time,
                        'CPU->GPU transfer': total_transfer_to_gpu_time,
                        'Model inference': total_inference_time,
                        'GPU->CPU transfer': total_transfer_to_cpu_time,
                        'Post-processing': total_postprocess_time
                    }
                    sorted_times = sorted(times.items(), key=lambda x: x[1], reverse=True)
                    for i, (name, t) in enumerate(sorted_times[:3]):
                        print(f'{i+1}. {name}: {t:.3f}s ({t/total_measured*100:.1f}%)')
                print('='*60 + '\n')

                print(f"End of Testing! Args: {args}, Input image: {img_list[N]}, Output shape: {output_img_shape}, Time cost: {time_cost} s, FPS: {fps:.2f}")

            torch.cuda.synchronize()
            nvtx.range_pop()

            mean_ssim_score, std_ssim_score, min_ssim_score, max_ssim_score, scores = 0, 0, 0, 0, 0
            if args.ssim:
                mean_ssim_score, std_ssim_score, min_ssim_score, max_ssim_score, scores = calculate_ssim_with_256(output_img, args)
                print(f"Mean SSIM score: {mean_ssim_score:.6f}, Std SSIM score: {std_ssim_score:.6f}, Min SSIM score: {min_ssim_score:.6f}, Max SSIM score: {max_ssim_score:.6f}")
                print(np.sort(scores)[:24])
                print(np.argsort(scores)[:24])

            with open("log_results.txt", "a") as f:
                f.write(f"Args: {args}, Input image: {img_list[N]}, Output shape: {output_img_shape}, Time cost: {time_cost} s, FPS: {fps:.2f}, SSIM score: {mean_ssim_score:.6f}, Std SSIM score: {std_ssim_score:.6f}, Min SSIM score: {min_ssim_score:.6f}, Max SSIM score: {max_ssim_score:.6f}\n")
            
            if args.fake_quant:
                with open("best_scale_dict.json", "w") as f:
                    json.dump(best_scale_dict, f, indent=4)

if __name__ == '__main__':
    program_start_time = time.time()
    parser = argparse.ArgumentParser(description='Prepear for testing')

    # Training parameters
    parser.add_argument("--GPU", type=str, default='0', help="The index of GPU you will use for computation (e.g. '0', '0,1')")
    parser.add_argument("--patch_xy", type=int, default=256, help="Patch size in x and y")
    parser.add_argument("--burst", type=int, default=8, help="The burst size in t")
    parser.add_argument('--overlap_factor', type=float, default=0.25, help="The overlap factor between two adjacent patches")
    parser.add_argument('--datasets_folder', type=str, default='20251211_CX3CR1_high', help="The folder containing files for testing")
    parser.add_argument('--denoise_model', type=str, default='20251211_CX3CR1_high_202601060823', help='The folder containing models to be tested')

    parser.add_argument("--datasets_path", type=str, default='./datasets', help="The root path of testing dataset")
    parser.add_argument("--pth_path", type=str, default='./pth', help="The root path to save models")
    parser.add_argument("--output_path", type=str, default='./results', help="Output path")

    parser.add_argument("--num_workers", type=int, default=4, help="4 for Linux system and 0 for Windows system")
    parser.add_argument('--scale_factor', type=int, default=1, help='The factor for image intensity scaling')

    parser.add_argument("--no_save", action='store_true', default=False, help="Whether to save the output image")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for inference (default: number of GPUs). Larger batch size can improve GPU utilization.")
    parser.add_argument("--use_compile", action='store_true', default=False, help="Use torch.compile() to accelerate inference (PyTorch 2.0+, recommended)")
    parser.add_argument("--use_jit", action='store_true', default=False, help="Use TorchScript JIT to accelerate inference (fallback if compile not available)")
    parser.add_argument("--ssim", action='store_true', default=False, help="Compare with GT")
    parser.add_argument("--GT_path", type=str, default=None, help="The root path of GT")
    parser.add_argument("--fake_quant", action='store_true', default=False, help="Use fake quantization to accelerate inference")
    parser.add_argument("--exclude_layers", type=str, nargs="+", default=[], help="Layer name substrings to exclude from quantization")
    args = parser.parse_args()

    if args.GT_path is not None:
        assert os.path.exists(args.GT_path), "GT path does not exist"
        assert args.GT_path.endswith('.tif'), "GT path must be a .tif file"

    test(args)
    program_end_time = time.time()
    program_execution_time = program_end_time - program_start_time
    print(f"Program execution time: {program_execution_time:.2f} seconds")
