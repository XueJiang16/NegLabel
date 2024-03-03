import os.path as osp
import pickle
import random
import shutil
import tempfile
import time

import mmcv
import numpy
import numpy as np
import torch
import torch.distributed as dist
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def single_gpu_test_ood(model,
                        data_loader,
                        name=''
                        ):
    """Test model with local single gpu.

    This method tests model with a single gpu and supports showing results.

    Args:
        model (:obj:`torch.nn.Module`): Model to be tested.
        data_loader (:obj:`torch.utils.data.DataLoader`): Pytorch data loader.
        show (bool): Whether to show the test results. Defaults to False.
        out_dir (str): The output directory of result plots of all samples.
            Defaults to None, which means not to write output files.
        **show_kwargs: Any other keyword arguments for showing results.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog = 0
        tic = time.time()
        # prog_bar = mmcv.ProgressBar(len(dataset))
    if world_size > 1:
        dist.barrier()
    for i, data in enumerate(data_loader):
        data['dataset_name'] = name
        result = model.forward(**data)
        if len(result.shape) == 0:  # handle the situation of batch = 1
            result = result.unsqueeze(0)
        results.append(result)
        # results.append(result.unsqueeze(0))  #############
        if rank == 0:
            batch_size = data['img'].size(0)
            prog += batch_size * world_size
            toc = time.time()
            passed_time = toc - tic
            inf_speed = passed_time / prog
            fps = 1 / inf_speed
            eta = max(0, (len(dataset) - prog)) * inf_speed
            print("[{} @ {}] {} / {}, fps = {}, eta = {}"
                  .format(name, int(passed_time), prog, len(dataset), round(fps, 2), round(eta, 2)))
    if world_size > 1:
        dist.barrier()
    results = torch.cat(results).cpu().numpy()
    # results = results.mean(0)
    # if rank == 0:
    #     time.sleep(2)
    # print(results.cpu().tolist())
    # time.sleep(10)
    # assert False
    return results

def ssim_test(img, img_metas=None, **kwargs):
    crop_size = 120
    img_size = 480
    crops = []
    crops_mean = []
    crops_std = []
    corner_list = []
    for h in range(img_size//crop_size):
        for w in range(img_size//crop_size):
            corner_list.append([h*crop_size, w*crop_size])
    for h,w in corner_list:
        crop = img[:, :, h:h+crop_size, w:w+crop_size]
        std, mean = torch.std_mean(crop, dim=(1,2,3))
        crops_mean.append(mean.unsqueeze(1))
        crops_std.append(std.unsqueeze(1))
    crops_mean = torch.cat(crops_mean, dim=1)
    crops_std = torch.cat(crops_std, dim=1)
    ssim_crops = torch.std(crops_mean, dim=1) + 3 * torch.std(crops_std, dim=1)
    return ssim_crops


def single_gpu_test_ssim(model,
                         data_loader,
                         name=''
                         ):
    """Test model with local single gpu.

    This method tests model with a single gpu and supports showing results.

    Args:
        model (:obj:`torch.nn.Module`): Model to be tested.
        data_loader (:obj:`torch.utils.data.DataLoader`): Pytorch data loader.
        show (bool): Whether to show the test results. Defaults to False.
        out_dir (str): The output directory of result plots of all samples.
            Defaults to None, which means not to write output files.
        **show_kwargs: Any other keyword arguments for showing results.

    Returns:
        list: The prediction results.
    """
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog = 0
        tic = time.time()
        # prog_bar = mmcv.ProgressBar(len(dataset))
    if world_size > 1:
        dist.barrier()
    for i, data in enumerate(data_loader):
        data['dataset_name'] = name
        result = ssim_test(**data)
        result = torch.tensor(result).to('cuda:{}'.format(rank))
        if len(result.shape) == 0:  # handle the situation of batch = 1
            result = result.unsqueeze(0)
        results.append(result)
        if rank == 0:
            batch_size = data['img'].size(0)
            prog += batch_size * world_size
            toc = time.time()
            passed_time = toc - tic
            inf_speed = passed_time / prog
            fps = 1 / inf_speed
            eta = max(0, (len(dataset) - prog)) * inf_speed
            print("[{} @ {}] {} / {}, fps = {}, eta = {}"
                  .format(name, int(passed_time), prog, len(dataset), round(fps, 2), round(eta, 2)))
    if world_size > 1:
        dist.barrier()
    results = torch.cat(results).cpu().numpy()
    return results


def single_gpu_test_ood_score(model,
                              data_loader,
                              name=''
                              ):
    """Test model with local single gpu.

    This method tests model with a single gpu and supports showing results.

    Args:
        model (:obj:`torch.nn.Module`): Model to be tested.
        data_loader (:obj:`torch.utils.data.DataLoader`): Pytorch data loader.
        show (bool): Whether to show the test results. Defaults to False.
        out_dir (str): The output directory of result plots of all samples.
            Defaults to None, which means not to write output files.
        **show_kwargs: Any other keyword arguments for showing results.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        cat_scores = []
        prog = 0
        tic = time.time()
        # prog_bar = mmcv.ProgressBar(len(dataset))
    if world_size > 1:
        dist.barrier()
    for i, data in enumerate(data_loader):
        result, cat_score = model.forward(**data)
        if len(result.shape) == 0:  # handle the situation of batch = 1
            result = result.unsqueeze(0)
        results.append(result)
        if rank == 0:
            cat_scores.append(cat_score)
            batch_size = data['img'].size(0)
            prog += batch_size * world_size
            toc = time.time()
            passed_time = toc - tic
            inf_speed = passed_time / prog
            fps = 1 / inf_speed
            eta = max(0, (len(dataset) - prog)) * inf_speed
            print("[{} @ {}] {} / {}, fps = {}, eta = {}"
                  .format(name, int(passed_time), prog, len(dataset), round(fps, 2), round(eta, 2)))
    if world_size > 1:
        dist.barrier()
    if rank == 0:
        x = np.arange(1, 11, 1)
    results = torch.cat(results).cpu().numpy()
    return results, results
