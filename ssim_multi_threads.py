from skimage.metrics import structural_similarity as ssim
import numpy as np
from multiprocessing import Pool, cpu_count

def ssim_single_slice(args):
    img1, img2 = args
    return ssim(
        img1,
        img2,
        data_range=img2.max() - img2.min()
    )

def ssim_volume_parallel(output_img, gt_img, num_workers=None):
    if num_workers is None:
        num_workers = cpu_count()//2

    with Pool(num_workers) as pool:
        scores = pool.map(
            ssim_single_slice,
            zip(output_img, gt_img)
        )
    
    std_scores = np.std(scores)
    mean_scores = float(np.mean(scores))
    min_scores = float(np.min(scores))
    max_scores = float(np.max(scores))
    return mean_scores, std_scores, min_scores, max_scores, scores