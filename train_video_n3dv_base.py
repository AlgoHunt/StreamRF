from hashlib import md5
from multiprocessing import process
from operator import index
from pydoc import describe
import torch
import torch.cuda
import torch.optim
import torch.nn as nn
import torch.nn.functional as F


import svox2
import svox2.csrc as _C
import svox2.utils
import json
import imageio
import os
from os import path
import time
import shutil
import gc
import math
import argparse

import numpy as np

from util.dataset import datasets
from util.util import Timing, get_expon_lr_func, viridis_cmap
from util import config_util

from warnings import warn
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from typing import NamedTuple, Optional, Union
from loguru import logger
from multiprocess import Pool


# runtime_svox2file = os.path.join(os.path.dirname(svox2.__file__), 'svox2.py')
# update_svox2file = '../svox2/svox2.py'
# if md5(open(runtime_svox2file,'rb').read()).hexdigest() != md5(open(update_svox2file,'rb').read()).hexdigest():
#     raise Exception("Not INSTALL the NEWEST svox2.py")

device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
config_util.define_common_args(parser)

group = parser.add_argument_group("general")
group.add_argument('--train_dir', '-t', type=str, default='ckpt',
                     help='checkpoint and logging directory')
group.add_argument('--basis_type',
                    choices=['sh', '3d_texture', 'mlp'],
                    default='sh',
                    help='Basis function type')
group.add_argument('--sh_dim', type=int, default=9, help='SH/learned basis dimensions (at most 10)')

group = parser.add_argument_group("optimization")
group.add_argument('--n_iters', type=int, default=10 * 12800, help='total number of iters to optimize for')
group.add_argument('--batch_size', type=int, default=
                     20000,
                   help='batch size')
group.add_argument('--sigma_optim', choices=['sgd', 'rmsprop'], default='rmsprop', help="Density optimizer")
group.add_argument('--lr_sigma', type=float, default=3e1, help='SGD/rmsprop lr for sigma')
group.add_argument('--lr_sigma_final', type=float, default=5e-2)
group.add_argument('--lr_sigma_decay_steps', type=int, default=250000)
group.add_argument('--lr_sigma_delay_steps', type=int, default=15000,
                   help="Reverse cosine steps (0 means disable)")
group.add_argument('--lr_sigma_delay_mult', type=float, default=1e-2)#1e-4)#1e-4)


group.add_argument('--sh_optim', choices=['sgd', 'rmsprop'], default='rmsprop', help="SH optimizer")
group.add_argument('--lr_sh', type=float, default=1e-2,help='SGD/rmsprop lr for SH')
group.add_argument('--lr_sh_final', type=float,default=5e-6)
group.add_argument('--lr_sh_decay_steps', type=int, default=250000)
group.add_argument('--lr_sh_delay_steps', type=int, default=0, help="Reverse cosine steps (0 means disable)")
group.add_argument('--lr_sh_delay_mult', type=float, default=1e-2)

group.add_argument('--lr_fg_begin_step', type=int, default=0, help="Foreground begins training at given step number")

group.add_argument('--rms_beta', type=float, default=0.95, help="RMSProp exponential averaging factor")

group.add_argument('--print_every', type=int, default=20, help='print every')
group.add_argument('--save_every', type=int, default=5,
                   help='save every x epochs')
group.add_argument('--eval_every', type=int, default=1,
                   help='evaluate every x epochs')

group.add_argument('--init_sigma', type=float,
                   default=0.1,
                   help='initialization sigma')
group.add_argument('--log_mse_image', action='store_true', default=False)
group.add_argument('--log_depth_map', action='store_true', default=False)
group.add_argument('--log_depth_map_use_thresh', type=float, default=None,
        help="If specified, uses the Dex-neRF version of depth with given thresh; else returns expected term")


group = parser.add_argument_group("misc experiments")
group.add_argument('--thresh_type',
                    choices=["weight", "sigma"],
                    default="weight",
                   help='Upsample threshold type')
group.add_argument('--weight_thresh', type=float,
                    default=0.0005 * 512,
                    #  default=0.025 * 512,
                   help='Upsample weight threshold; will be divided by resulting z-resolution')
group.add_argument('--density_thresh', type=float,
                    default=5.0,
                   help='Upsample sigma threshold')
group.add_argument('--background_density_thresh', type=float,
                    default=1.0+1e-9,
                   help='Background sigma threshold for sparsification')
group.add_argument('--max_grid_elements', type=int,
                    default=44_000_000,
                   help='Max items to store after upsampling '
                        '(the number here is given for 22GB memory)')



group = parser.add_argument_group("losses")
# Foreground TV
group.add_argument('--lambda_tv', type=float, default=1e-5)
group.add_argument('--tv_sparsity', type=float, default=0.01)
group.add_argument('--tv_logalpha', action='store_true', default=False,
                   help='Use log(1-exp(-delta * sigma)) as in neural volumes')

group.add_argument('--lambda_tv_sh', type=float, default=1e-3)
group.add_argument('--tv_sh_sparsity', type=float, default=0.01)

group.add_argument('--lambda_tv_lumisphere', type=float, default=0.0)#1e-2)#1e-3)
group.add_argument('--tv_lumisphere_sparsity', type=float, default=0.01)
group.add_argument('--tv_lumisphere_dir_factor', type=float, default=0.0)

group.add_argument('--tv_decay', type=float, default=1.0)

group.add_argument('--lambda_l2_sh', type=float, default=0.0)#1e-4)
group.add_argument('--tv_early_only', type=int, default=1, help="Turn off TV regularization after the first split/prune")

group.add_argument('--tv_contiguous', type=int, default=1,
                        help="Apply TV only on contiguous link chunks, which is faster")
# End Foreground TV

group.add_argument('--lr_decay', action='store_true', default=True)
group.add_argument('--n_train', type=int, default=None, help='Number of training images. Defaults to use all avaiable.')


group.add_argument('--lambda_sparsity', type=float, default=
                    0.0,
                    help="Weight for sparsity loss as in SNeRG/PlenOctrees " +
                         "(but applied on the ray)")
group.add_argument('--lambda_beta', type=float, default=
                    0.0,
                    help="Weight for beta distribution sparsity loss as in neural volumes")

# ---------------- Finetune video related--------------
group = parser.add_argument_group("finetune")
group.add_argument('--pretrained', type=str, default=None,
                    help='pretrained model')

group.add_argument('--mask_grad_after_reg', type=int, default=1,
                    help='mask out unwanted gradient after TV and other regularization')

group.add_argument('--frame_start', type=int, default=1, help='train frame among [frame_start, frame_end]')  
group.add_argument('--frame_end', type=int, default=30, help='train frame among [1, frame_end]')
group.add_argument('--fps', type=int, default=30, help='video save fps')

group.add_argument('--train_use_all', type=int, default=0 ,help='whether to use all image as training set')
group.add_argument('--save_every_frame', action='store_true', default=False)
group.add_argument('--dilate_rate_before', type=int, default=2, help="dilation rate for grid.links before training")
group.add_argument('--dilate_rate_after', type=int, default=2, help=" dilation rate for grid.links after training")


group.add_argument('--offset', type=int, default=250)

# fancy idea
group.add_argument('--compress_saving', action="store_true", default=False, help="dilation rate for grid.links")
group.add_argument('--sh_keep_thres', type=float, default=1)
group.add_argument('--sh_prune_thres', type=float, default=0.2)

group.add_argument('--performance_mode', action="store_true", default=False, help="use perfomance_mode skip any unecessary code ")
group.add_argument('--debug',  action="store_true", default=False,help="switch on debug mode")
group.add_argument('--keep_rms_data',  action="store_true", default=False,help="switch on debug mode")


group.add_argument('--apply_narrow_band',  action="store_true", default=False,help="apply_narrow_band")
group.add_argument('--render_all',  action="store_true", default=False,help="render all camera in sequence")
group.add_argument('--save_delta',  action="store_true", default=False,help="save delta in compress saving")

args = parser.parse_args()
config_util.maybe_merge_config_file(args)

DEBUG = args.debug
assert args.lr_sigma_final <= args.lr_sigma, "lr_sigma must be >= lr_sigma_final"
assert args.lr_sh_final <= args.lr_sh, "lr_sh must be >= lr_sh_final"

os.makedirs(args.train_dir, exist_ok=True)
os.makedirs(os.path.join(args.train_dir, 'grid_delta'), exist_ok=True)
os.makedirs(os.path.join(args.train_dir, 'grid_delta_z'), exist_ok=True)
os.makedirs(os.path.join(args.train_dir, 'test_images'), exist_ok=True)
os.makedirs(os.path.join(args.train_dir, 'test_images_depth'), exist_ok=True)

logfolder = args.train_dir
if os.path.exists(f'{logfolder}/log_base.log'):
    os.remove(f'{logfolder}/log_base.log')
logger.add(f'{logfolder}/log_base.log' , format="{level} {message}", level='DEBUG' if args.debug else 'INFO')

summary_writer = SummaryWriter(args.train_dir)

with open(path.join(args.train_dir, 'args.json'), 'w') as f:
    json.dump(args.__dict__, f, indent=2)
    # Changed name to prevent errors
    shutil.copyfile(__file__, path.join(args.train_dir, 'train_frozen.py'))

torch.manual_seed(20200823)
np.random.seed(20200823)

assert os.path.exists(args.pretrained), "pretrained model not exist, please train the first frame!"
print("Load pretrained model from ", args.pretrained)
grid = svox2.SparseGrid.load(args.pretrained, device=device)
config_util.setup_render_opts(grid.opt, args)
print("Load pretrained model Done!")

from copy import deepcopy
from torch import nn




def grid_copy( old_grid: svox2.SparseGrid, device: Union[torch.device, str] = "cpu"):
    """
    Load from path
    """

    sh_data = old_grid.sh_data.clone()
    density_data = old_grid.density_data.clone()
    logger.debug(f"copy grid cap {(old_grid.links>=0).sum()}")
    if hasattr(old_grid, "background_links") :
        background_data = old_grid.background_data
        background_links = old_grid.background_links
    else:
        background_data = None
        background_links = None
        
    links = old_grid.links.clone()
    basis_dim = (sh_data.shape[1]) // 3
    radius = deepcopy(old_grid.radius )
    center = deepcopy(old_grid.center)
    grid_new = svox2.SparseGrid(
        1,
        radius=radius,
        center=center,
        basis_dim=basis_dim,
        use_z_order=False,
        device="cpu",
        basis_type=old_grid.basis_type ,
        mlp_posenc_size=old_grid.mlp_posenc_size, 
        mlp_width=old_grid.mlp_width,
        background_nlayers=0,
    )
   
    grid_new.sh_data = nn.Parameter(sh_data).to(device=device)
    grid_new.density_data = nn.Parameter(density_data).to(device=device)
    grid_new.links = links.to(device=device) # torch.from_numpy(links).to(device=device)
    grid_new.capacity = grid_new.sh_data.size(0)
    if args.keep_rms_data:
        grid_new.sh_rms = old_grid.sh_rms
        grid_new.density_rms = old_grid.density_rms

    if background_data is not None:
        background_data = torch.from_numpy(background_data).to(device=device)
        grid_new.background_nlayers = background_data.shape[1]
        grid_new.background_reso = background_links.shape[1]
        grid_new.background_data = nn.Parameter(background_data)
        grid_new.background_links = torch.from_numpy(background_links).to(device=device)
    else:
        grid_new.background_data.data = grid_new.background_data.data.to(device=device)

    if grid_new.links.is_cuda:
        grid_new.accelerate()
    config_util.setup_render_opts(grid_new.opt, args)
    logger.debug(f"grid copy finish")
    return grid_new

def delete_area(grid, delet_mask):
    new_mask = torch.logical_and(grid.links>=0, ~delet_mask)

    delet_mask = delet_mask[grid.links>=0]
    grid.density_data = nn.Parameter(grid.density_data[~delet_mask,:])
    grid.sh_data = nn.Parameter(grid.sh_data[~delet_mask,:])
    if args.keep_rms_data:
        grid.sh_rms = None 
        grid.density_rms = None 

    new_links = torch.cumsum(new_mask.view(-1).to(torch.int32), dim=-1).int() - 1
    new_links[~new_mask.view(-1)] = -1
    grid.links = new_links.view(grid.links.shape)

@torch.no_grad()
def compress_saving(grid_pre, grid_next, grid_holder, save_delta=False,saving_name=None):
    mask_pre = grid_pre.links>=0
    mask_next = grid_next.links>=0
    new_cap = mask_next.sum()
    
    diff_area = torch.logical_xor(mask_pre, mask_next)

    add_area   =  (diff_area & mask_next)
    minus_area =  (diff_area & mask_pre)
    
    addition_density = grid_next.density_data[grid_next.links[add_area].long()]
    addition_sh = grid_next.sh_data[grid_next.links[add_area].long()]

    logger.debug(f"diff area: {diff_area.sum()} add area: {add_area.sum()} minus area: {minus_area.sum()} ")
    remain_idx = grid_pre.links[mask_pre & ~ minus_area]
    remain_idx = remain_idx.long()

    remain_sh_data = grid_pre.sh_data[remain_idx]
    remain_density_data = grid_pre.density_data[remain_idx]

    new_sh_data = torch.zeros((new_cap,27), device=device).float()
    new_density_data = torch.zeros((new_cap,1), device=device).float()
    
    add_area_in_saprse = add_area[mask_next]
    
    # we also save voxel where sh change a lot
    next_sh_data = grid_next.sh_data[~add_area_in_saprse,:]
    next_density_data = grid_next.density_data[~add_area_in_saprse,:]
    part2_keep_area = (abs(next_sh_data - remain_sh_data).sum(-1) > args.sh_keep_thres)
    keep_numel = part2_keep_area.sum()
    add_numel = add_area.sum()
    
    keep_percent = (keep_numel/new_cap) * 100
    add_percent = (add_numel/new_cap) * 100
    keep_size = (keep_numel*2*28)/(1024*1024) 
    add_size = (add_numel*2*28)/(1024*1024)
    if save_delta:
        save_dict = {'mask_next':mask_next,
                    'addition_density':addition_density,
                    'addition_sh':addition_sh,
                    'part2_keep_area':part2_keep_area,
                    'keep_density':next_density_data[part2_keep_area],
                    'keep_sh':next_sh_data[part2_keep_area]
        }
        save_path = os.path.join(args.train_dir,'grid_delta',f'{saving_name}.pth')
        logger.info(f'svaing delta to : {save_path} ')
        torch.save(save_dict, save_path)
        logger.info(f"keep element: {keep_numel}/{keep_percent:.2f}/{keep_size:.2f} MB, add element: {add_numel}/{add_percent:.2f}/{add_size:.2f} MB")

    if save_delta:
        all_in_one =  {
            'mask_next':np.packbits(mask_next.cpu().numpy()),
            'mask_keep':np.packbits(part2_keep_area.cpu().numpy()) ,
            'addition_density':addition_density.cpu().numpy().astype(np.float16),
            'addition_sh':addition_sh.cpu().numpy().astype(np.float16),
            'keep_density':next_density_data[part2_keep_area].cpu().numpy().astype(np.float16),
            'keep_sh':next_sh_data[part2_keep_area].cpu().numpy().astype(np.float16)
        }
        save_path = os.path.join(args.train_dir,'grid_delta_z',f'{saving_name}.npz')
        
        np.savez_compressed(save_path, all_in_one)
        logger.info(f'saving delta z to : {save_path}') 
        logger.info(f'saving size after compression: {os.path.getsize(save_path)/(1024*1024):.2f} MB')

    remain_sh_data[part2_keep_area] = next_sh_data[part2_keep_area]
    remain_density_data[part2_keep_area] = next_density_data[part2_keep_area]

    new_sh_data[add_area_in_saprse,:] = addition_sh 
    new_density_data[add_area_in_saprse,:] = addition_density 
    new_sh_data[~add_area_in_saprse,:] = remain_sh_data
    new_density_data[~add_area_in_saprse,:] = remain_density_data
    # though new_links equal to grid_next.links, we still calculate a mask for better scalability
    new_mask = torch.logical_or(add_area, mask_pre)
    new_mask = torch.logical_and(new_mask, ~minus_area)
    new_links = torch.cumsum(new_mask.view(-1).to(torch.int32), dim=-1).int() - 1
    new_links[~new_mask.view(-1)] = -1
    

    grid_holder.sh_data = nn.Parameter(new_sh_data)
    grid_holder.density_data = nn.Parameter(new_density_data)
    grid_holder.links = new_links.view(grid_next.links.shape).to(device=device)

    if args.keep_rms_data:
        grid_holder.sh_rms = grid_next.sh_rms
        grid_holder.density_rms = grid_next.density_rms
    
    logger.debug(f"compress saving finish")

    return  grid_holder


def dilated_voxel_grid(dilate_rate = 2):
    active_mask = grid.links >= 0
    dilate_before = active_mask
    for i in range(dilate_rate):
        active_mask = _C.dilate(active_mask)
    # reactivate = torch.logical_xor(active_mask, dilate_before)
    new_cap = active_mask.sum()
    previous_sparse_area = dilate_before[active_mask]
    
    new_density = torch.zeros((new_cap,1), device=device).float()
    new_sh = torch.zeros((new_cap, grid.basis_dim*3),  device=device).float()
    
    new_density[previous_sparse_area,:] = grid.density_data.data
    new_sh[previous_sparse_area,:] = grid.sh_data.data

    active_mask = active_mask.view(-1)
    new_links = torch.cumsum(active_mask.to(torch.int32), dim=-1).int() - 1
    new_links[~active_mask] = -1

    grid.density_data = torch.nn.Parameter(new_density)
    grid.sh_data = torch.nn.Parameter(new_sh)
    grid.links =  new_links.view(grid.links.shape).to(device=device)

   
def sparsify_voxel_grid(grid, factor=[1,1,1],dilate=2):
    reso = grid.links.shape
    reso = [int(r * fac) for r, fac in zip(reso, factor)]
    grid.resample(reso=reso,
                    sigma_thresh=args.density_thresh,
                    weight_thresh=0.0,
                    dilate=dilate, 
                    cameras= None,
                    max_elements=args.max_grid_elements,
                    accelerate=False)



if args.dilate_rate_after > 0:
    logger.debug("sparsify first!!!!")
    sparsify_voxel_grid(grid,dilate=args.dilate_rate_after)


# LR related
lr_sigma_func = get_expon_lr_func(args.lr_sigma, args.lr_sigma_final, args.lr_sigma_delay_steps,
                                  args.lr_sigma_delay_mult, args.lr_sigma_decay_steps)
lr_sh_func = get_expon_lr_func(args.lr_sh, args.lr_sh_final, args.lr_sh_delay_steps,
                               args.lr_sh_delay_mult, args.lr_sh_decay_steps)
lr_sigma_factor = 1.0
lr_sh_factor = 1.0



grid_raw = grid_copy(grid, device=device)


from torch.multiprocessing  import Queue, Process
from queue import Empty
frame_idx_queue = Queue()
dset_queue = Queue()

def pre_fetch_dataset():
    while True:
        try:
            frame_idx = frame_idx_queue.get(block=True,timeout=60)
        except Empty:
            logger.debug('ending data prefetch process')
            return 
        data_dir = os.path.join(args.data_dir, f'{frame_idx:04d}')
        train_dir = args.train_dir
        factor = 1
        dset_train = datasets[args.dataset_type](
                    data_dir,
                    split="train",
                    device=device,
                    factor=factor,
                    n_images=args.n_train,
                    train_dir = train_dir,
                    train_use_all=args.train_use_all,
                    offset=args.offset,
                    verbose=False,
                    **config_util.build_data_options(args))
        
        # dataset used to render test image, can include training camera for better visualization 
        dset_test = datasets[args.dataset_type](
                    data_dir, split= 'train' if args.render_all else "test", train_use_all=1 if args.render_all else 0,offset=args.offset, verbose=False, **config_util.build_data_options(args))

        # # dataset used for PSNR caculation 
        dset_eval = datasets[args.dataset_type](
                    data_dir, split="test", train_use_all=0,offset=args.offset, verbose=False, **config_util.build_data_options(args))

        logger.debug(f"finish loading frame:{frame_idx}")
        dset_queue.put((dset_train,dset_test, dset_eval))
    return dset_train, dset_test, dset_eval

def pre_fetch_dataset_standalone(frame_idx):
    data_dir = os.path.join(args.data_dir, f'{frame_idx:04d}')
    train_dir = args.train_dir
    factor = 1
    dset_train = datasets[args.dataset_type](
                data_dir,
                split="train",
                device=device,
                factor=factor,
                n_images=args.n_train,
                train_dir = train_dir,
                train_use_all=args.train_use_all,
                offset=args.offset,
                verbose=False,
                **config_util.build_data_options(args))
    
    # dataset used to render test image, can include training camera for better visualization 
    dset_test = datasets[args.dataset_type](
                data_dir, split= 'train' if args.render_all else "test", train_use_all=1 if args.render_all else 0,offset=args.offset, verbose=False, **config_util.build_data_options(args))

    # # dataset used for PSNR caculation 
    dset_eval = datasets[args.dataset_type](
                data_dir, split="test", train_use_all=0,offset=args.offset, verbose=False, **config_util.build_data_options(args))

    logger.debug(f"finish loading frame:{frame_idx}")
    return dset_train, dset_test, dset_eval



def deploy_dset(dset):
    dset.c2w = torch.from_numpy(dset.c2w)
    dset.gt = torch.from_numpy(dset.gt).float()
    if not dset.is_train_split:
        dset.render_c2w = torch.from_numpy(dset.render_c2w)
    else:
        dset.gen_rays()  
    return dset



def finetune_one_frame(frame_idx, global_step_base, dsets):
    if args.compress_saving:
        grid_pre = grid_copy(old_grid = grid, device=device)
    with torch.no_grad():
        if args.apply_narrow_band:
            active_mask = grid.links>= 0
            dmask = active_mask.clone()
            for _ in range(args.dilate_rate_before):  
                dmask = _C.dilate(dmask)
            emask =  ~active_mask
            for _ in range(6):  
                emask = _C.dilate(emask)
            emask = ~emask
            narrow_band = torch.logical_xor(dmask, emask)


    if args.dilate_rate_before > 0:
        dilated_voxel_grid(dilate_rate=args.dilate_rate_before)

    if args.apply_narrow_band:
        grad_mask = narrow_band[grid.links>=0]
        grad_mask = grad_mask.view(-1)
    else:
        grad_mask = (torch.ones([1]).float().cuda() == 1)
   

    train_dir = args.train_dir
    
    dset_train, dset_test, dset_eval = dsets
    dset_train = deploy_dset(dset_train)
    dset_test  = deploy_dset(dset_test)
    dset_eval  = deploy_dset(dset_eval)

    epoch_id = -1
    global_start_time = datetime.now()
    gstep_id_base = 0
    

    shuffle_step = args.n_iters 
    dset_train.epoch_size = shuffle_step * args.batch_size
   

    timer_dict = {'forward':0, 'regularization':0, 'optimization':0,'preparation':0,'narrowband':0}
    max_step = args.n_iters * (10 if frame_idx == 0 else 1)
    grid.accelerate()
    for gstep_id in tqdm(range(0, max_step)):
        if gstep_id==0 or gstep_id % shuffle_step == 0:
            with torch.no_grad():
                dset_train.shuffle_rays()
                logger.debug('shuffle')

        def train_step(timer_dict):

            #============================= ray preparation stage =============================
            tic = time.time()
            stats = {"mse" : 0.0, "psnr" : 0.0, "invsqr_mse" : 0.0}

            bstep_id = gstep_id % shuffle_step
            batch_begin = bstep_id * args.batch_size 

            lr_sigma = lr_sigma_func(gstep_id) * lr_sigma_factor
            lr_sh = lr_sh_func(gstep_id) * lr_sh_factor
            if not args.lr_decay:
                lr_sigma = args.lr_sigma * lr_sigma_factor
                lr_sh = args.lr_sh * lr_sh_factor
            

            batch_end = batch_begin + args.batch_size 
            batch_origins = dset_train.rays.origins[batch_begin: batch_end]
            batch_dirs = dset_train.rays.dirs[batch_begin: batch_end]
            
            rgb_gt = dset_train.rays.gt[batch_begin: batch_end]
            rays = svox2.Rays(batch_origins, batch_dirs)
            if args.debug:
                torch.cuda.synchronize()
            timer_dict['preparation'] += (time.time() - tic)
    

            #============================= forward stage =============================
            tic = time.time()
            
            rgb_pred = grid.volume_render_fused(rays, rgb_gt,
                    beta_loss=args.lambda_beta,
                    sparsity_loss=args.lambda_sparsity,
                    randomize=args.enable_random)
                
            if args.debug:
                torch.cuda.synchronize()
            timer_dict['forward'] += (time.time() - tic)

            if not args.performance_mode:
                with torch.no_grad():
                    mse = F.mse_loss(rgb_gt, rgb_pred)
                mse_num : float = mse.detach().item()
                psnr = -10.0 * math.log10(mse_num)
                stats['mse'] += mse_num
                stats['psnr'] += psnr
                stats['invsqr_mse'] += 1.0 / mse_num ** 2

                if (gstep_id + 1) % args.print_every == 0:
                
                    for stat_name in stats:
                        stat_val = stats[stat_name] / args.print_every
                        summary_writer.add_scalar(stat_name, stat_val, global_step=gstep_id+global_step_base)
                        stats[stat_name] = 0.0
                    summary_writer.add_scalar("lr_sh", lr_sh, global_step=gstep_id+global_step_base)
                    summary_writer.add_scalar("lr_sigma", lr_sigma, global_step=gstep_id+global_step_base)
            
            #============================= regularization stage =============================
            tic = time.time()
            # Apply TV/Sparsity regularizers 
            if args.lambda_tv > 0.0:
                #  with Timing("tv_inpl"):
                grid.inplace_tv_grad(grid.density_data.grad,
                        scaling=args.lambda_tv,
                        sparse_frac=args.tv_sparsity ,
                        logalpha=args.tv_logalpha,
                        ndc_coeffs=dset_train.ndc_coeffs,
                        contiguous=args.tv_contiguous)

            if args.lambda_tv_sh > 0.0:
                #  with Timing("tv_color_inpl"):
                grid.inplace_tv_color_grad(grid.sh_data.grad,
                        scaling=args.lambda_tv_sh,
                        sparse_frac=args.tv_sh_sparsity,
                        ndc_coeffs=dset_train.ndc_coeffs,
                        contiguous=args.tv_contiguous)

            if args.lambda_tv_lumisphere > 0.0:
                grid.inplace_tv_lumisphere_grad(grid.sh_data.grad,
                        scaling=args.lambda_tv_lumisphere,
                        dir_factor=args.tv_lumisphere_dir_factor,
                        sparse_frac=args.tv_lumisphere_sparsity,
                        ndc_coeffs=dset_train.ndc_coeffs)
            if args.lambda_l2_sh > 0.0:
                grid.inplace_l2_color_grad(grid.sh_data.grad,
                        scaling=args.lambda_l2_sh)
    
            if args.debug:
                torch.cuda.synchronize()
            timer_dict['regularization']  += (time.time() - tic)
           

            #============================= narrow band stage =============================
            tic = time.time()
            grid.sparse_sh_grad_indexer &= grad_mask
            grid.sparse_grad_indexer &= grad_mask
            if args.debug:
                torch.cuda.synchronize()
                timer_dict['narrowband'] += (time.time() - tic)

            #============================= optimization stage =============================
            tic = time.time()
            grid.optim_density_step(lr_sigma, beta=args.rms_beta, optim=args.sigma_optim)
            grid.optim_sh_step(lr_sh, beta=args.rms_beta, optim=args.sh_optim)
            if args.debug:
                torch.cuda.synchronize()
            timer_dict['optimization'] += (time.time() - tic)
            
        def eval_step():
            with torch.no_grad():
                stats_test = {'mse' : 0.0, 'psnr' : 0.0}
                # Standard set
                N_IMGS_TO_EVAL = min(20 if epoch_id > 0 else 5, dset_eval.n_images)
                N_IMGS_TO_SAVE = N_IMGS_TO_EVAL # if not args.tune_mode else 1
                img_eval_interval = dset_eval.n_images // N_IMGS_TO_EVAL
                img_save_interval = (N_IMGS_TO_EVAL // N_IMGS_TO_SAVE)
                img_ids = range(0, dset_eval.n_images, img_eval_interval)
                n_images_gen = 0
                for i, img_id in tqdm(enumerate(img_ids), total=len(img_ids)):
                    c2w = dset_eval.c2w[img_id].to(device=device)
                    cam = svox2.Camera(c2w,
                                    dset_eval.intrins.get('fx', img_id),
                                    dset_eval.intrins.get('fy', img_id),
                                    dset_eval.intrins.get('cx', img_id),
                                    dset_eval.intrins.get('cy', img_id),
                                    width=dset_eval.get_image_size(img_id)[1],
                                    height=dset_eval.get_image_size(img_id)[0],
                                    ndc_coeffs=dset_eval.ndc_coeffs)
                    rgb_pred_test = grid.volume_render_image(cam, use_kernel=True)
                    rgb_gt_test = dset_eval.gt[img_id].to(device=device)
                    all_mses = ((rgb_gt_test - rgb_pred_test) ** 2).cpu()
                    if i % img_save_interval == 0:
                        img_pred = rgb_pred_test.cpu()
                        img_pred.clamp_max_(1.0)
                        summary_writer.add_image(f'test/image_{img_id:04d}',
                                img_pred, global_step=frame_idx, dataformats='HWC')
                        if args.log_mse_image:
                            mse_img = all_mses / all_mses.max()
                            summary_writer.add_image(f'test/mse_map_{img_id:04d}',
                                    mse_img, global_step=frame_idx, dataformats='HWC')
                        if False or args.log_depth_map:
                            depth_img = grid.volume_render_depth_image(cam,
                                        args.log_depth_map_use_thresh if
                                        args.log_depth_map_use_thresh else None
                                    )
                            depth_img = viridis_cmap(depth_img.cpu())
                            summary_writer.add_image(f'test/depth_map_{img_id:04d}',
                                    depth_img,
                                    global_step=frame_idx, dataformats='HWC')

                    rgb_pred_test = rgb_gt_test = None
                    mse_num : float = all_mses.mean().item()
                    psnr = -10.0 * math.log10(mse_num)
                    if math.isnan(psnr):
                        print('NAN PSNR', i, img_id, mse_num)
                        assert False
                    stats_test['mse'] += mse_num
                    stats_test['psnr'] += psnr
                    n_images_gen += 1
                stats_test['mse'] /= n_images_gen
                stats_test['psnr'] /= n_images_gen
                for stat_name in stats_test:
                    summary_writer.add_scalar('test/' + stat_name,
                            stats_test[stat_name], global_step=gstep_id_base+global_step_base)
                summary_writer.add_scalar('epoch_id', float(epoch_id), global_step=gstep_id_base+global_step_base)
                print('eval stats:', stats_test)
                logger.critical(f"per_frame_psnr: {frame_idx}  {psnr}")
                return psnr
        if args.debug:
            torch.cuda.synchronize()
        
        tic = time.time()
        train_step(timer_dict)
        if args.debug:
            torch.cuda.synchronize()

        if gstep_id == max_step - 1:
            global_stop_time = datetime.now()
            line = ''
            for k,v in timer_dict.items():
                line += f'{k}:sum: {v:.3f} sec / avg:{(v*1000)/max_step:.3f} ms,  '
            logger.info(line)
            secs = (global_stop_time - global_start_time).total_seconds()
            logger.info(f'cost: {secs},  s')
            psnr = eval_step()
            
            break


    if args.dilate_rate_after or args.dilate_rate_before:
     
        sparsify_voxel_grid(grid, dilate=args.dilate_rate_after)
   
    @torch.no_grad()
    def preprune(grid_pre, grid_next):
    
        mask_pre = grid_pre.links>=0
        mask_next = grid_next.links>=0
        new_cap = mask_next.sum()
        
        diff_area = torch.logical_xor(mask_pre, mask_next)

        add_area   =  (diff_area & mask_next)
        minus_area =  (diff_area & mask_pre)
        logger.info(f"diff area before preprune: {diff_area.sum()} add area: {add_area.sum()} minus area: {minus_area.sum()} ")
        addition_density = grid_next.density_data[grid_next.links[add_area].long()]
        addition_sh = grid_next.sh_data[grid_next.links[add_area].long()]

        no_need_area = (abs(addition_sh).sum(-1)<args.sh_prune_thres)
        add_area[add_area.clone()] = no_need_area
        delete_area(grid_next, add_area.view(grid_next.links.shape))
        
    if args.compress_saving:
        preprune(grid_pre, grid)
        compress_saving(grid_pre=grid_pre, grid_next=grid, grid_holder=grid, save_delta=args.save_delta,saving_name=f'{frame_idx:04d}')
    
    def render_img():
        c2ws = dset_test.c2w.to(device=device)
        
        n_images = dset_test.n_images
        img_eval_interval = 1
        for img_id in tqdm(range(0, n_images, img_eval_interval)):

            dset_h, dset_w = dset_test.get_image_size(img_id)
            im_size = dset_h * dset_w
            w = dset_w #if args.crop == 1.0 else int(dset_w * args.crop)
            h = dset_h #if args.crop == 1.0 else int(dset_h * args.crop)
            
            if args.render_all:
                im_path = os.path.join(train_dir, 'test_images', f'{frame_idx:04d}_{img_id:02d}.png' )
                depth_path = os.path.join(train_dir, 'test_images_depth', f'{frame_idx:04d}_{img_id:02d}.png' )
            else:
                im_path = os.path.join(train_dir, 'test_images', f'{frame_idx:04d}.png' )
            cam = svox2.Camera(c2ws[img_id],
                            dset_test.intrins.get('fx', img_id),
                            dset_test.intrins.get('fy', img_id),
                            dset_test.intrins.get('cx', img_id) + (w - dset_w) * 0.5,
                            dset_test.intrins.get('cy', img_id) + (h - dset_h) * 0.5,
                            w, h,
                            ndc_coeffs=dset_test.ndc_coeffs)

            tic = time.time()
            im = grid.volume_render_image(cam, use_kernel=True, return_raylen=False)
            if DEBUG:
                torch.cuda.synchronize()
                logger.debug(f'rgb rendeing time: {time.time() - tic}')
            im.clamp_(0.0, 1.0)
            im = im.cpu().numpy()
            im = (im * 255).astype(np.uint8)
            imageio.imwrite(im_path, im)
            if not args.render_all:
                break

        return im
    with torch.no_grad():
        return render_img(), psnr


train_start_time = datetime.now()
train_frame_num = 0
global_step_base = 0
frames = []
psnr_list = []

pre_fetch_process = Process(target=pre_fetch_dataset)

pre_fetch_process.start()
prefetch_factor = 3
for i in range(prefetch_factor):
    frame_idx_queue.put(i+args.frame_start)


for frame_idx in range(args.frame_start, args.frame_end) :
    # dset = dset_iter[frame_idx - args.frame_start]
   
    dset = dset_queue.get(block=True)

    if frame_idx + prefetch_factor < args.frame_end:
        frame_idx_queue.put(frame_idx + prefetch_factor)

    frame, psnr = finetune_one_frame(frame_idx, global_step_base, dset)
    frames.append(frame)
    psnr_list.append(psnr)
    if args.save_every_frame:
        os.makedirs(os.path.join(args.train_dir,"ckpts"))
        grid.save(os.path.join(args.train_dir,"ckpts",f'{frame_idx:04d}.npz'))

    global_step_base += args.n_iters
    train_frame_num += 1

logger.critical(f'average psnr {sum(psnr_list)/len(psnr_list):.4f}')



if train_frame_num:

    tag = os.path.basename(args.train_dir)
    vid_path = os.path.join(args.train_dir, tag+'.mp4')
    # dep_vid_path = os.path.join(args.train_dir, 'render_depth.mp4')
    imageio.mimwrite(vid_path, frames, fps=args.fps, macro_block_size=8)
    logger.info('video write to', vid_path)

    grid.density_rms = torch.zeros([1])
    grid.sh_rms = torch.zeros([1])
    grid.save(os.path.join(args.train_dir, 'ckpt.npz'))

pre_fetch_process.join()
pre_fetch_process.close()