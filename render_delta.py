from hashlib import md5
from operator import index
from pydoc import describe
import torch
import torch.cuda
import torch.optim
import torch.nn.functional as F
import svox2
import svox2.csrc as _C
import svox2.utils
import json
import imageio
import os
from os import path
import shutil
import gc
import numpy as np
import math
import argparse
from util.dataset import datasets
from util.util import Timing, get_expon_lr_func, viridis_cmap
from util import config_util

from warnings import warn
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from typing import NamedTuple, Optional, Union
from loguru import logger
import time

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

# group.add_argument('--tune_mode', action='store_true', default=False,
#                    help='hypertuning mode (do not save, for speed)')
group.add_argument('--tune_nosave', action='store_true', default=False,
                   help='do not save any checkpoint even at the end')
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
group.add_argument('--strategy', type=int, default=0,
                    help='specfic sample startegy')
group.add_argument('--mask_grad_after_reg', type=int, default=1,
                    help='mask out unwanted gradient after TV and other regularization')
group.add_argument('--view_count_thres', type=int, default=-1,
                    help='mask out unwanted gradient after TV and other regularization')

group.add_argument('--frame_start', type=int, default=1, help='train frame among [frame_start, frame_end]')  
group.add_argument('--frame_end', type=int, default=30, help='train frame among [1, frame_end]')
group.add_argument('--fps', type=int, default=30, help='video save fps')
group.add_argument('--save_every_frame', action='store_true', default=False)
group.add_argument('--dilate_rate', type=int, default=2, help="dilation rate for grid.links")
group.add_argument('--use_grad_mask', action="store_true", default=False, help="dilation rate for grid.links")
group.add_argument('--offset', type=int, default=250)

group.add_argument('--sh_keep_thres', type=float, default=1)
group.add_argument('--performance_mode', action="store_true", default=False, help="use perfomance_mode skip any unecessary code ")
group.add_argument('--debug',  action="store_true", default=False,help="switch on debug mode")
group.add_argument('--keep_rms_data',  action="store_true", default=False,help="switch on debug mode")


group.add_argument('--render_all',  action="store_true", default=False,help="render all camera in sequence")


args = parser.parse_args()
config_util.maybe_merge_config_file(args)

DEBUG = args.debug
assert args.lr_sigma_final <= args.lr_sigma, "lr_sigma must be >= lr_sigma_final"
assert args.lr_sh_final <= args.lr_sh, "lr_sh must be >= lr_sh_final"

os.makedirs(args.train_dir, exist_ok=True)
os.makedirs(os.path.join(args.train_dir, 'test_images_sc'), exist_ok=True)
os.makedirs(os.path.join(args.train_dir, 'test_images_depth_sc'), exist_ok=True)

summary_writer = SummaryWriter(args.train_dir)

with open(path.join(args.train_dir, 'args.json'), 'w') as f:
    json.dump(args.__dict__, f, indent=2)
    # Changed name to prevent errors
    shutil.copyfile(__file__, path.join(args.train_dir, 'opt_frozen.py'))

torch.manual_seed(20200823)
np.random.seed(20200823)

assert os.path.exists(args.pretrained), "pretrained model not exist, please train the first frame!"
print("Load pretrained model from ", args.pretrained)
grid = svox2.SparseGrid.load(args.pretrained, device=device)
config_util.setup_render_opts(grid.opt, args)
print("Load pretrained model Done!")

from copy import deepcopy
from torch import nn


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


def grid_copy( old_grid: svox2.SparseGrid, device: Union[torch.device, str] = "cpu"):
    """
    Load from path
    """

    sh_data = old_grid.sh_data.clone()
    density_data = old_grid.density_data.clone()
    logger.error(f"copy grid cap {(old_grid.links>=0).sum()}")
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
   
    grid_new.viewcount_helper = torch.zeros_like(density_data, dtype=torch.int, device=device)
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
    return grid_new

from torch import nn


def compress_loading(grid_pre, delta_path):

    delta  = torch.load(delta_path)

    mask_next = delta['mask_next'].to(device)
    addition_density = delta['addition_density'].to(device)
    addition_sh = delta['addition_sh'].to(device)
    keep_density = delta['keep_density'].to(device)
    keep_sh = delta['keep_sh'].to(device)
    part2_keep_area =delta['part2_keep_area'] .to(device)

    mask_pre = grid_pre.links>=0
    new_cap = mask_next.sum()
    diff_area = torch.logical_xor(mask_pre, mask_next)
  
    add_area   =  (diff_area & mask_next)
    minus_area =  (diff_area & mask_pre)
    
  
    (abs(addition_sh).sum(-1)<=0.9).sum()
    # import ipdb;ipdb.set_trace()

    logger.debug(f"diff area: {diff_area.sum()} add area: {add_area.sum()} minus area: {minus_area.sum()} ")
    remain_idx = grid_pre.links[mask_pre & ~ minus_area]
    remain_idx = remain_idx.long()

    remain_sh_data = grid_pre.sh_data[remain_idx]
    remain_density_data = grid_pre.density_data[remain_idx]

    
    new_sh_data = torch.zeros((new_cap,27), device=device).float()
    new_density_data = torch.zeros((new_cap,1), device=device).float()
    
    add_area_in_saprse = add_area[mask_next]
    
    # we also save voxel where sh change a lot


    # import ipdb;ipdb.set_trace()
    keep_numel = part2_keep_area.sum()
    add_numel = add_area.sum()
    
    keep_percent = (keep_numel/new_cap) * 100
    add_percent = (add_numel/new_cap) * 100
    keep_size = (keep_numel*2*28)/(1024*1024) 
    add_size = (add_numel*2*28)/(1024*1024)
    
    logger.info(f"keep element: {keep_numel}/{keep_percent:.2f}/{keep_size:.2f} MB,  add element: {add_numel}/{add_percent:.2f}/{add_size:.2f} MB")

    remain_sh_data[part2_keep_area] = keep_sh
    remain_density_data[part2_keep_area] = keep_density

    new_sh_data[add_area_in_saprse,:] = addition_sh 
    new_density_data[add_area_in_saprse,:] = addition_density 
    new_sh_data[~add_area_in_saprse,:] = remain_sh_data
    new_density_data[~add_area_in_saprse,:] = remain_density_data
    
    # though new_links equal to grid_next.links, we still calculate a mask for better code scalability
    new_mask = torch.logical_or(add_area, mask_pre)
    new_mask = torch.logical_and(new_mask, ~minus_area)
    new_links = torch.cumsum(new_mask.view(-1).to(torch.int32), dim=-1).int() - 1
    new_links[~new_mask.view(-1)] = -1
    
    # import ipdb;ipdb.set_trace()
    grid_pre.sh_data = nn.Parameter(new_sh_data)
    grid_pre.density_data = nn.Parameter(new_density_data)
    grid_pre.links = new_links.view(grid_pre.links.shape).to(device=device)

    return  grid_pre



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



def sparsify_voxel_grid():
    reso = grid.links.shape
    grid.resample(reso=reso,
                    sigma_thresh=args.density_thresh,
                    weight_thresh=0.0,
                    dilate=2, 
                    cameras= None,
                    max_elements=args.max_grid_elements,
                    accelerate=False)

def sparsify_voxel_grid_fast(dilate=2):
    reso = grid.links.shape
    sample_vals_mask = grid.density_data >= args.density_thresh
    max_elements=args.max_grid_elements
    if max_elements > 0 and max_elements < grid.density_data.numel() \
                        and max_elements < torch.count_nonzero(sample_vals_mask):
        # To bound the memory usage
        sigma_thresh_bounded = torch.topk(grid.density_data.view(-1),
                            k=max_elements, sorted=False).values.min().item()
        sigma_thresh = max(sigma_thresh, sigma_thresh_bounded)
        print(' Readjusted sigma thresh to fit to memory:', sigma_thresh)
        sample_vals_mask = grid.density_data >= sigma_thresh
    if grid.opt.last_sample_opaque:
        # Don't delete the last z layer
        sample_vals_mask[:, :, -1] = 1
    if dilate:
        for i in range(int(dilate)):
            sample_vals_mask = _C.dilate(sample_vals_mask)
    sample_vals_density = grid.density_data[sample_vals_mask]
    cnz = torch.count_nonzero(sample_vals_mask).item()
    sample_vals_sh = grid.sh_data[sample_vals_mask]
    init_links = (
                    torch.cumsum(sample_vals_mask.to(torch.int32), dim=-1).int() - 1
                )
    init_links[~sample_vals_mask] = -1
    grid.capacity = cnz
    print(" New cap:", grid.capacity)
    del sample_vals_mask
    print('density', sample_vals_density.shape, sample_vals_density.dtype)
    print('sh', sample_vals_sh.shape, sample_vals_sh.dtype)
    print('links', init_links.shape, init_links.dtype)
    grid.density_data = nn.Parameter(sample_vals_density.view(-1, 1).to(device=device))
    grid.sh_data = nn.Parameter(sample_vals_sh.to(device=device))
    grid.links = init_links.view(reso).to(device=device)

if args.dilate_rate > 0:
    logger.info("sparsify first!!!!")
    
    # grid_pre = grid_copy(grid, device=device)
    sparsify_voxel_grid()
    # slow_sparse_grid = grid_copy(grid, device=device)
    # grid = grid_pre
    # sparsify_voxel_grid()
    # import ipdb;ipdb.set_trace()

# LR related
lr_sigma_func = get_expon_lr_func(args.lr_sigma, args.lr_sigma_final, args.lr_sigma_delay_steps,
                                  args.lr_sigma_delay_mult, args.lr_sigma_decay_steps)
lr_sh_func = get_expon_lr_func(args.lr_sh, args.lr_sh_final, args.lr_sh_delay_steps,
                               args.lr_sh_delay_mult, args.lr_sh_decay_steps)
lr_sigma_factor = 1.0
lr_sh_factor = 1.0



grid_raw = grid_copy(grid, device=device)





def eval_one_frame(frame_idx, global_step_base):
  
    
    data_dir = os.path.join(args.data_dir, f'{frame_idx:04d}')
    train_dir = args.train_dir
    factor = 1

    dset_test = datasets[args.dataset_type](
                data_dir, split= 'train' if args.render_all else "test", train_use_all=1 if args.render_all else 0,offset=args.offset, **config_util.build_data_options(args))

    dset_eval = datasets[args.dataset_type](
                data_dir, split="test", train_use_all=0,offset=args.offset, **config_util.build_data_options(args))
                
    torch.save(dset_test.c2w,os.path.join(args.train_dir, 'c2w.pth'))
    epoch_id = -1

    gstep_id_base = 0
    
    # dset_motion.epoch_size = args.n_iters * args.batch_size
    # indexer_motion = dset_motion.shuffle_rays(strategy=strategy, replace=replacement)

    delta_path = os.path.join(args.train_dir,'grid_delta_full',f'{frame_idx:04d}.pth')
    compress_loading(grid_pre=grid, delta_path=delta_path)

    

    def eval_step():
            
        stats_test = {'mse' : 0.0, 'psnr' : 0.0}
        # Standard set
        N_IMGS_TO_EVAL = min(20 if epoch_id > 0 else 5, dset_eval.n_images)
        N_IMGS_TO_SAVE = N_IMGS_TO_EVAL # if not args.tune_mode else 1
        img_eval_interval = dset_eval.n_images // N_IMGS_TO_EVAL
        img_save_interval = (N_IMGS_TO_EVAL // N_IMGS_TO_SAVE)
        img_ids = range(0, dset_eval.n_images, img_eval_interval)
        n_images_gen = 0
        for i, img_id in tqdm(enumerate(img_ids), total=len(img_ids)):
            c2w = torch.from_numpy(dset_eval.c2w[img_id]).to(device=device)
            cam = svox2.Camera(c2w,
                            dset_eval.intrins.get('fx', img_id),
                            dset_eval.intrins.get('fy', img_id),
                            dset_eval.intrins.get('cx', img_id),
                            dset_eval.intrins.get('cy', img_id),
                            width=dset_eval.get_image_size(img_id)[1],
                            height=dset_eval.get_image_size(img_id)[0],
                            ndc_coeffs=dset_eval.ndc_coeffs)
            rgb_pred_test = grid.volume_render_image(cam, use_kernel=True)
            rgb_gt_test = torch.from_numpy(dset_eval.gt[img_id]).to(device=device)
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

        
    
    eval_step()
            
    
    def render_img():
       
        c2ws = torch.from_numpy(dset_test.c2w).to(device=device)
        
        n_images = dset_test.n_images
        img_eval_interval = 1
        for img_id in tqdm(range(0, n_images, img_eval_interval)):

            dset_h, dset_w = dset_test.get_image_size(img_id)
            im_size = dset_h * dset_w
            w = dset_w #if args.crop == 1.0 else int(dset_w * args.crop)
            h = dset_h #if args.crop == 1.0 else int(dset_h * args.crop)
            
            if args.render_all:
                im_path = os.path.join(train_dir, 'test_images_sc', f'{frame_idx:04d}_{img_id:02d}.png' )
                depth_path = os.path.join(train_dir, 'test_images_depth_sc', f'{frame_idx:04d}_{img_id:02d}.png' )
            else:
                im_path = os.path.join(train_dir, 'test_images_sc', f'{frame_idx:04d}.png' )
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
        eval_step()
        return render_img()



train_start_time = datetime.now()
train_frame_num = 0
global_step_base = 0
frames = []
for frame_idx in range(args.frame_start, args.frame_end):
    frames.append(eval_one_frame(frame_idx, global_step_base))
    global_step_base += args.n_iters
    train_frame_num += 1


train_end_time = datetime.now()
secs = (train_end_time-train_start_time).total_seconds()

if train_frame_num:
    average_time = secs / train_frame_num
    print(f'train {train_frame_num} images, cost {secs} s, average {average_time}s per image')
    tag = os.path.basename(args.train_dir)
    vid_path = os.path.join(args.train_dir, tag+'_from_saved_delta.mp4')
    # dep_vid_path = os.path.join(args.train_dir, 'render_depth.mp4')
    imageio.mimwrite(vid_path, frames, fps=args.fps, macro_block_size=8)
    print('video write into', vid_path)
    print('Final save ckpt')

