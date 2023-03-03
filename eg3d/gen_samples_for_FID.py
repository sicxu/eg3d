# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Generate images and shapes using pretrained network pickle."""

import os
import re
from typing import List, Optional, Tuple, Union

import click
import dnnlib
import numpy as np
import PIL.Image
import torch
from tqdm import tqdm
import mrcfile


import legacy
from camera_utils import LookAtPoseSampler, FOV_to_intrinsics
from torch_utils import misc
from training.triplane import TriPlaneGenerator
import imageio


def iterate_random_labels(dataset_kwargs, batch_size, c_dim, device):
    if c_dim == 0:
        c = torch.zeros([batch_size, c_dim], device=device)
        while True:
            yield c
    else:
        dataset = dnnlib.util.construct_class_by_name(**dataset_kwargs)
        while True:
            c = [dataset.get_label(np.random.randint(len(dataset))) for _i in range(batch_size)]
            c = torch.from_numpy(np.stack(c)).pin_memory().to(device)
            yield c

def iterate_random_pose(rendering_kwargs, batch_size, device='cpu'):
    while True:
        lookat_position = torch.tensor(rendering_kwargs['avg_camera_pivot'], dtype=torch.float32, device=device)
        lookat_position = lookat_position.unsqueeze(0).repeat(batch_size, 1)
        forward_cam2world_pose = LookAtPoseSampler.sample(
                    3.14/2, 3.14/2, 
                    lookat_position=lookat_position,
                    horizontal_stddev=rendering_kwargs['h_std'], 
                    vertical_stddev=rendering_kwargs['v_std'], 
                    radius=rendering_kwargs['avg_camera_radius'], 
                    batch_size=batch_size,
                    device=device)
                
        intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
        intrinsics = intrinsics.unsqueeze(0).repeat(batch_size, 1, 1)
        pose = torch.cat([forward_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
        yield pose


#----------------------------------------------------------------------------
@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seed', help='Random seed', metavar='INT', type=click.IntRange(min=0), default=0, show_default=True)
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--frames', help='', type=int, required=False, metavar='int', default=2048, show_default=True)
@click.option('--batch_size', help='', type=int, required=False, metavar='int', default=32, show_default=True)
@click.option('--data', help='Training data', metavar='[ZIP|DIR]', type=str, required=True)
@click.option('--dataset_name', help='dataset name', metavar='STR', type=str, default='ImageFolderDataset', required=False)


def generate_images(
    network_pkl: str,
    seed: int,
    frames: int,
    batch_size: int,
    outdir: str,
    data: str,
    dataset_name: str
):
    """Generate images using pretrained network pickle.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.%s'%dataset_name, path=data, use_labels=True, max_size=None, xflip=False)
    c_iter = iterate_random_labels(dataset_kwargs, batch_size=batch_size, c_dim=G.c_dim, device=device)
    pose_iter = iterate_random_pose(G.rendering_kwargs, batch_size=batch_size, device=device)
    outdir = os.path.join(outdir, f"sampled-{frames//1000}k-{network_pkl.split('/')[-1].split('.')[0]:s}")
    os.makedirs(outdir, exist_ok=True)

    # Generate images.
    for i in range(frames // batch_size):
        print('Generating image for batch %d...' % (i))
        z = torch.randn([batch_size, G.z_dim], device=device)
        ws = G.mapping(z, c=next(c_iter))
        img = G.synthesis(ws, pose=next(pose_iter))['image']
        img = (img.permute(0, 2, 3, 1)  * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        for j in range(batch_size):
            PIL.Image.fromarray(img[j].cpu().numpy(), 'RGB').save(f'{outdir}/seed{seed:04d}_{i*batch_size + j:05d}.png')

#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
