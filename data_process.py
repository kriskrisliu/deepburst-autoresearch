import numpy as np
import os
import tifffile as tiff
from skimage import io
import random
import math
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import time

def random_transform(input, target):
    p_trans = random.randrange(8)
    if p_trans == 0:  # no transformation
        input = input
        target = target
    elif p_trans == 1:  # left rotate 90
        input = np.rot90(input, k=1, axes=(1, 2))
        target = np.rot90(target, k=1, axes=(1, 2))
    elif p_trans == 2:  # left rotate 180
        input = np.rot90(input, k=2, axes=(1, 2))
        target = np.rot90(target, k=2, axes=(1, 2))
    elif p_trans == 3:  # left rotate 270
        input = np.rot90(input, k=3, axes=(1, 2))
        target = np.rot90(target, k=3, axes=(1, 2))
    elif p_trans == 4:  # horizontal flip
        input = input[:, :, ::-1]
        target = target[:, :, ::-1]
    elif p_trans == 5:  # horizontal flip & left rotate 90
        input = input[:, :, ::-1]
        input = np.rot90(input, k=1, axes=(1, 2))
        target = target[:, :, ::-1]
        target = np.rot90(target, k=1, axes=(1, 2))
    elif p_trans == 6:  # horizontal flip & left rotate 180
        input = input[:, :, ::-1]
        input = np.rot90(input, k=2, axes=(1, 2))
        target = target[:, :, ::-1]
        target = np.rot90(target, k=2, axes=(1, 2))
    elif p_trans == 7:  # horizontal flip & left rotate 270
        input = input[:, :, ::-1]
        input = np.rot90(input, k=3, axes=(1, 2))
        target = target[:, :, ::-1]
        target = np.rot90(target, k=3, axes=(1, 2))
    return input, target


class trainset(Dataset):
    def __init__(self, name_list, coordinate_list, noise_img_all, stack_index, burst_size):
        self.name_list = name_list
        self.coordinate_list = coordinate_list
        self.noise_img_all = noise_img_all
        self.stack_index = stack_index
        self.burst_size = burst_size

    def __getitem__(self, index):
        stack_index = self.stack_index[index]
        noise_img = self.noise_img_all[stack_index]
        single_coordinate = self.coordinate_list[self.name_list[index]]
        init_h = single_coordinate['init_h']
        end_h = single_coordinate['end_h']
        init_w = single_coordinate['init_w']
        end_w = single_coordinate['end_w']
        init_s = single_coordinate['init_s']
        end_s = single_coordinate['end_s']
        
        input = noise_img[init_s: init_s + self.burst_size, init_h: end_h, init_w: end_w]
        target = noise_img[init_s + self.burst_size: end_s, init_h: end_h, init_w: end_w]
        
        # input, target = random_transform(input, target)

        p_exc = random.random()  # generate a random number determinate whether swap input and target

        if p_exc < 0.5:
            input, target = random_transform(input, target)
        else:
            input, target = target, input
            input, target = random_transform(input, target)

        # input = torch.from_numpy(np.expand_dims(input, 0).copy())
        # target = torch.from_numpy(np.expand_dims(target, 0).copy())
        input = torch.from_numpy(input.copy())
        target = torch.from_numpy(target.copy())
        
        target = target.reshape(target.shape[0] // self.burst_size, self.burst_size, *target.shape[1:]).mean(dim=1)

        return input, target
    
    def __len__(self):
        return len(self.name_list)
    
    
class testset(Dataset):
    def __init__(self, name_list, coordinate_list, noise_img):
        self.name_list = name_list
        self.coordinate_list = coordinate_list
        self.noise_img = noise_img

    def __getitem__(self, index):
        single_coordinate = self.coordinate_list[self.name_list[index]]
        init_h = single_coordinate['init_h']
        end_h = single_coordinate['end_h']
        init_w = single_coordinate['init_w']
        end_w = single_coordinate['end_w']
        init_s = single_coordinate['init_s']
        end_s = single_coordinate['end_s']
        noise_patch = self.noise_img[init_s:end_s, init_h:end_h, init_w:end_w]
        noise_patch = torch.from_numpy(noise_patch)
        noise_patch = noise_patch.unsqueeze(1)

        return noise_patch, single_coordinate
    
    def __len__(self):
        return len(self.name_list)

def train_preprocess_lessMemoryMulStacks(args):
    patch_y = args.patch_y
    patch_x = args.patch_x
    patch_t = 2 * args.burst
    gap_y = args.gap_y
    gap_x = args.gap_x
    gap_t = args.burst
    im_folder = os.path.join(args.datasets_path, args.datasets_folder)

    name_list = []
    coordinate_list={}
    stack_index = []
    noise_im_all = []
    ind = 0
    print('\033[1;31mImage list for training -----> \033[0m')
    print('All files are in -----> ', im_folder)
    stack_num = len(list(os.walk(im_folder, topdown=False))[-1][-1])
    print('Total stack number -----> ', stack_num)

    print('Reading files...') 
    print('\033[1;33mPlease check the shape of these image stacks, since some hyperstacks have unusual shapes. In that case, you just need to re-store these images by ImageJ. \033[0m') 
    for im_name in list(os.walk(im_folder, topdown=False))[-1][-1]:
        im_dir = os.path.join(im_folder, im_name)
        noise_im = tiff.imread(im_dir)
        
        if args.train_datasets_size < noise_im.shape[0]:
            noise_im = noise_im[:args.train_datasets_size]
        
        print(im_name, ' -----> the shape is', noise_im.shape)

        noise_im = noise_im.astype(np.float32) / args.scale_factor  # no preprocessing
        # noise_im = noise_im-noise_im.mean()
        noise_im = (noise_im-noise_im.mean())/(noise_im.std())

        noise_im_all.append(noise_im)
        
        whole_x = noise_im.shape[2]
        whole_y = noise_im.shape[1]
        whole_t = noise_im.shape[0]
                
        num_h = math.ceil((whole_y - patch_y + gap_y) / gap_y)
        num_w = math.ceil((whole_x - patch_x + gap_x) / gap_x)
        num_s = math.ceil((whole_t - patch_t + gap_t) / gap_t)
    
        for x in range(0, num_h):
            for y in range(0, num_w):
                for z in range(0, num_s):
                    single_coordinate = {'init_h': 0, 'end_h': 0, 'init_w': 0, 'end_w': 0, 'init_s': 0, 'end_s': 0}
                    
                    if x != (num_h - 1):
                        init_h = gap_y * x
                        end_h = gap_y * x + patch_y
                    elif x == (num_h - 1):
                        init_h = whole_y - patch_y
                        end_h = whole_y

                    if y != (num_w - 1):
                        init_w = gap_x * y
                        end_w = gap_x * y + patch_x
                    elif y == (num_w - 1):
                        init_w = whole_x - patch_x
                        end_w = whole_x

                    init_s = gap_t * z
                    end_s = gap_t * z + patch_t
                    
                    single_coordinate['init_h'] = init_h
                    single_coordinate['end_h'] = end_h
                    single_coordinate['init_w'] = init_w
                    single_coordinate['end_w'] = end_w
                    single_coordinate['init_s'] = init_s
                    single_coordinate['end_s'] = end_s
                    patch_name = args.datasets_folder+'_'+im_name.replace('.tif','')+'_x'+str(x)+'_y'+str(y)+'_z'+str(z)
                    name_list.append(patch_name)
                    coordinate_list[patch_name] = single_coordinate
                    stack_index.append(ind)
        ind = ind + 1
    return name_list, noise_im_all, coordinate_list, stack_index


def test_preprocess_chooseOne(args, N):
    # 性能分析：开始计时
    total_start = time.time()
    
    # 1. 参数初始化和文件列表获取
    t1_start = time.time()
    patch_y = args.patch_y
    patch_x = args.patch_x
    patch_t2 = args.burst
    gap_y = args.gap_y
    gap_x = args.gap_x
    gap_t2 = args.burst
    cut_w = (patch_x - gap_x) / 2
    cut_h = (patch_y - gap_y) / 2
    cut_s = (patch_t2 - gap_t2) / 2
    im_folder = os.path.join(args.datasets_path, args.datasets_folder)

    name_list = []
    coordinate_list = {}
    img_list = list(os.walk(im_folder, topdown=False))[-1][-1]
    img_list.sort()

    im_name = img_list[N]
    t1_end = time.time()
    print(f"[性能分析] 1. 参数初始化和文件列表获取: {t1_end - t1_start:.4f} 秒")
    
    # 2. 图像读取
    t2_start = time.time()
    im_dir = os.path.join(im_folder, im_name)
    # noise_im = tiff.imread(im_dir)
    arrays = []
    with tiff.TiffFile(im_dir) as tif:
        for series in tqdm(tif.series, desc=f"Reading {im_name}"):
            arrays.append(series.asarray())

    noise_im = np.concatenate(arrays, axis=0)
    t2_end = time.time()
    print(f"[性能分析] 2. 图像读取 (TiffFile读取+concatenate): {t2_end - t2_start:.4f} 秒")
    
    # 3. 图像切片
    t3_start = time.time()
    noise_im = noise_im[int(0*noise_im.shape[0]):]
    t3_end = time.time()
    print(f"[性能分析] 3. 图像切片: {t3_end - t3_start:.4f} 秒")
    
    # 4. 统计信息计算（优化：先转换为float32再计算，加速大数组统计）
    t4_start = time.time()
    input_data_type = noise_im.dtype
    # 优化策略：如果原数组是更大的数据类型（如float64），先转换为float32
    # float32计算比float64快约2倍，且对于图像数据精度足够
    if noise_im.dtype != np.float32:
        noise_im_f32 = noise_im.astype(np.float32, copy=False)
        img_mean = noise_im_f32.mean()
        img_std = noise_im_f32.std(ddof=0)  # ddof=0计算总体标准差
    else:
        # 如果已经是float32，直接计算
        img_mean = noise_im.mean()
        img_std = noise_im.std(ddof=0)
        noise_im_f32 = None  # 标记未创建临时数组
    t4_end = time.time()
    print(f"[性能分析] 4. 统计信息计算 (mean+std): {t4_end - t4_start:.4f} 秒")

    # 5. 图像预处理（优化：复用已转换的float32数组，避免重复转换）
    t5_start = time.time()
    if noise_im_f32 is not None:
        # 复用已转换的float32数组，避免重复转换
        if args.scale_factor ==1:
            noise_im = noise_im_f32
        else:
            noise_im = noise_im_f32 / args.scale_factor
    else:
        if args.scale_factor ==1:
            noise_im = noise_im
        else:
            # 如果原数组已经是float32，直接使用（不需要再次转换）
            noise_im = noise_im / args.scale_factor
    noise_im = (noise_im - img_mean) / img_std
    t5_end = time.time()
    print(f"[性能分析] 5. 图像预处理 (类型转换+归一化): {t5_end - t5_start:.4f} 秒")

    # 6. 尺寸计算
    t6_start = time.time()
    whole_x = noise_im.shape[2]
    whole_y = noise_im.shape[1]
    whole_t = noise_im.shape[0]
    
    num_w = math.ceil((whole_x - patch_x + gap_x) / gap_x)
    num_h = math.ceil((whole_y - patch_y + gap_y) / gap_y)
    num_s = math.ceil((whole_t - patch_t2 + gap_t2) / gap_t2)
    t6_end = time.time()
    print(f"[性能分析] 6. 尺寸计算: {t6_end - t6_start:.4f} 秒")
    
    # 7. 三重循环生成坐标
    t7_start = time.time()
    for x in range(0, num_h):
        for y in range(0, num_w):
            for z in range(0, num_s):
                single_coordinate = {'init_h': 0, 'end_h': 0, 'init_w': 0, 'end_w': 0, 'init_s': 0, 'end_s': 0}
                if x != (num_h - 1):
                    init_h = gap_y * x
                    end_h = gap_y * x + patch_y
                elif x == (num_h - 1):
                    init_h = whole_y - patch_y
                    end_h = whole_y

                if y != (num_w - 1):
                    init_w = gap_x * y
                    end_w = gap_x * y + patch_x
                elif y == (num_w - 1):
                    init_w = whole_x - patch_x
                    end_w = whole_x

                if z != (num_s - 1):
                    init_s = gap_t2 * z
                    end_s = gap_t2 * z + patch_t2
                elif z == (num_s - 1):
                    init_s = whole_t - patch_t2
                    end_s = whole_t
                single_coordinate['init_h'] = init_h
                single_coordinate['end_h'] = end_h
                single_coordinate['init_w'] = init_w
                single_coordinate['end_w'] = end_w
                single_coordinate['init_s'] = init_s
                single_coordinate['end_s'] = end_s

                if y == 0:
                    single_coordinate['stack_start_w'] = y * gap_x
                    single_coordinate['stack_end_w'] = y * gap_x + patch_x - cut_w
                    single_coordinate['patch_start_w'] = 0
                    single_coordinate['patch_end_w'] = patch_x - cut_w
                elif y == num_w - 1:
                    single_coordinate['stack_start_w'] = whole_x - patch_x + cut_w
                    single_coordinate['stack_end_w'] = whole_x
                    single_coordinate['patch_start_w'] = cut_w
                    single_coordinate['patch_end_w'] = patch_x
                else:
                    single_coordinate['stack_start_w'] = y * gap_x + cut_w
                    single_coordinate['stack_end_w'] = y * gap_x + patch_x - cut_w
                    single_coordinate['patch_start_w'] = cut_w
                    single_coordinate['patch_end_w'] = patch_x - cut_w

                if x == 0:
                    single_coordinate['stack_start_h'] = x * gap_y
                    single_coordinate['stack_end_h'] = x * gap_y + patch_y - cut_h
                    single_coordinate['patch_start_h'] = 0
                    single_coordinate['patch_end_h'] = patch_y - cut_h
                elif x == num_h - 1:
                    single_coordinate['stack_start_h'] = whole_y - patch_y + cut_h
                    single_coordinate['stack_end_h'] = whole_y
                    single_coordinate['patch_start_h'] = cut_h
                    single_coordinate['patch_end_h'] = patch_y
                else:
                    single_coordinate['stack_start_h'] = x * gap_y + cut_h
                    single_coordinate['stack_end_h'] = x * gap_y + patch_y - cut_h
                    single_coordinate['patch_start_h'] = cut_h
                    single_coordinate['patch_end_h'] = patch_y - cut_h

                if z == 0:
                    single_coordinate['stack_start_s'] = z * gap_t2
                    single_coordinate['stack_end_s'] = z * gap_t2 + patch_t2 - cut_s
                    single_coordinate['patch_start_s'] = 0
                    single_coordinate['patch_end_s'] = patch_t2 - cut_s
                elif z == num_s - 1:
                    single_coordinate['stack_start_s'] = whole_t - patch_t2 + cut_s
                    single_coordinate['stack_end_s'] = whole_t
                    single_coordinate['patch_start_s'] = cut_s
                    single_coordinate['patch_end_s'] = patch_t2
                else:
                    single_coordinate['stack_start_s'] = z * gap_t2 + cut_s
                    single_coordinate['stack_end_s'] = z * gap_t2 + patch_t2 - cut_s
                    single_coordinate['patch_start_s'] = cut_s
                    single_coordinate['patch_end_s'] = patch_t2 - cut_s

                patch_name = args.datasets_folder+'_x'+str(x)+'_y'+str(y)+'_z'+str(z)
                name_list.append(patch_name)
                coordinate_list[patch_name] = single_coordinate
    t7_end = time.time()
    print(f"[性能分析] 7. 三重循环生成坐标: {t7_end - t7_start:.4f} 秒")
    
    # 性能分析：总时间
    total_end = time.time()
    print(f"[性能分析] ===== 总执行时间: {total_end - total_start:.4f} 秒 =====")
    print(f"[性能分析] 各部分时间占比:")
    total_time = total_end - total_start
    print(f"  1. 参数初始化和文件列表获取: {(t1_end - t1_start) / total_time * 100:.2f}%")
    print(f"  2. 图像读取: {(t2_end - t2_start) / total_time * 100:.2f}%")
    print(f"  3. 图像切片: {(t3_end - t3_start) / total_time * 100:.2f}%")
    print(f"  4. 统计信息计算: {(t4_end - t4_start) / total_time * 100:.2f}%")
    print(f"  5. 图像预处理: {(t5_end - t5_start) / total_time * 100:.2f}%")
    print(f"  6. 尺寸计算: {(t6_end - t6_start) / total_time * 100:.2f}%")
    print(f"  7. 三重循环生成坐标: {(t7_end - t7_start) / total_time * 100:.2f}%")

    return name_list, noise_im, coordinate_list, im_name, img_mean, input_data_type, img_std

def singlebatch_test_save(args, single_coordinate, output_image):
    stack_start_w = int(single_coordinate['stack_start_w'])
    stack_end_w = int(single_coordinate['stack_end_w'])
    patch_start_w = int(single_coordinate['patch_start_w'])
    patch_end_w = int(single_coordinate['patch_end_w'])

    stack_start_h = int(single_coordinate['stack_start_h'])
    stack_end_h = int(single_coordinate['stack_end_h'])
    patch_start_h = int(single_coordinate['patch_start_h'])
    patch_end_h = int(single_coordinate['patch_end_h'])

    stack_start_s = int(single_coordinate['stack_start_s'])
    stack_end_s = int(single_coordinate['stack_end_s'])
    patch_start_s = int(single_coordinate['patch_start_s'])
    patch_end_s = int(single_coordinate['patch_end_s'])

    output_patch = output_image[int(patch_start_s / args.burst): int(patch_end_s / args.burst), patch_start_h: patch_end_h, patch_start_w: patch_end_w]
    # raw_patch = raw_image[int((patch_start_s / args.burst)): int(patch_end_s / args.burst), patch_start_h: patch_end_h, patch_start_w: patch_end_w]
    return output_patch, stack_start_w, stack_end_w, stack_start_h, stack_end_h, stack_start_s, stack_end_s

def multibatch_test_save(args, single_coordinate, id, output_image):
    stack_start_w_id = single_coordinate['stack_start_w']#.numpy()
    stack_start_w = int(stack_start_w_id[id])
    stack_end_w_id = single_coordinate['stack_end_w']#.numpy()
    stack_end_w = int(stack_end_w_id[id])
    patch_start_w_id = single_coordinate['patch_start_w']#.numpy()
    patch_start_w = int(patch_start_w_id[id])
    patch_end_w_id = single_coordinate['patch_end_w']#.numpy()
    patch_end_w = int(patch_end_w_id[id])

    stack_start_h_id = single_coordinate['stack_start_h']#.numpy()
    stack_start_h = int(stack_start_h_id[id])
    stack_end_h_id = single_coordinate['stack_end_h']#.numpy()
    stack_end_h = int(stack_end_h_id[id])
    patch_start_h_id = single_coordinate['patch_start_h']#.numpy()
    patch_start_h = int(patch_start_h_id[id])
    patch_end_h_id = single_coordinate['patch_end_h']#.numpy()
    patch_end_h = int(patch_end_h_id[id])

    stack_start_s_id = single_coordinate['stack_start_s']#.numpy()
    stack_start_s = int(stack_start_s_id[id])
    stack_end_s_id = single_coordinate['stack_end_s']#.numpy()
    stack_end_s = int(stack_end_s_id[id])
    patch_start_s_id = single_coordinate['patch_start_s']#.numpy()
    patch_start_s = int(patch_start_s_id[id])
    patch_end_s_id = single_coordinate['patch_end_s']#.numpy()
    patch_end_s = int(patch_end_s_id[id])

    output_image_id = output_image[id:id+1]
    # raw_image_id = raw_image[id]
    output_patch = output_image_id[int((patch_start_s / args.burst)): int(patch_end_s / args.burst), patch_start_h: patch_end_h, patch_start_w: patch_end_w]
    # raw_patch = raw_image_id[int((patch_start_s / args.burst)): int(patch_end_s / args.burst), patch_start_h: patch_end_h, patch_start_w: patch_end_w]

    return output_patch, stack_start_w, stack_end_w, stack_start_h, stack_end_h, stack_start_s, stack_end_s
