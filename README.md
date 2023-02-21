# Streaming Radiance Fields for 3D Video Synthesis

Lingzhi Li, Zhen Shen, Zhongshu Wang, Li Shen, Ping Tan

Alibaba Group

Citation:
```

@inproceedings{li2022streaming,
  title={Streaming Radiance Fields for 3D Video Synthesis},
  author={Li, Lingzhi and Shen, Zhen and Shen, Li and Tan, Ping and others},
  booktitle={Advances in Neural Information Processing Systems}
}

```

arXiv: <https://arxiv.org/abs/2210.14831>



https://user-images.githubusercontent.com/28325733/210695784-a309dce8-533b-4c93-b637-da369e2a288e.mp4

Due to size limit, this is a downsampled video, check full resolution video [here](https://github.com/AlgoHunt/VideoHolder/releases/download/StreamRF/StreamRF-Camera.Ready.Video.mp4).

## Dataset
**Meet Room Dataset**: https://drive.google.com/drive/folders/1lNmQ6_ykyKjT6UKy-SnqWoSlI5yjh3l_?usp=share_link


**N3DV Dataset**:
https://github.com/facebookresearch/Neural_3D_Video




## Training StreamRF

Following the [setup](https://github.com/sxyu/svox2#setup) of the orginal plenoxels' repository 

For each scene, extract frames from every video, and arrange them into the following structure:

```
<data_dir>
  ├── 0000  
  |   ├── pose_bounds.npy  
  |   └── images
  |       └── cam[00/01/02/.../20].png
  ...
  └── 0299 
      ├── pose_bounds.npy  
      └── images
          └── cam[00/01/02/.../20].png
```
We provide the pose_bounds.npy of both dataset in the Meet Room Dataset's [link]( https://drive.google.com/drive/folders/1lNmQ6_ykyKjT6UKy-SnqWoSlI5yjh3l_?usp=share_link). If you wants to generate poses_bounds.npy for yourself check DS-NeRF's [repo](https://github.com/dunbar12138/DSNeRF#generate-camera-poses-and-sparse-depth-information-using-colmap-optional).

### Meet Room Dataset

1. Initialize the first frame model

```bash
python opt.py  -t <log_dir> <data_dir>/0000 -c configs/meetroom_init.json --scale 1.0
```

2. Train the pilot model

```bash
python train_video_n3dv_pilot.py -t <log_dir> <data_dir> -c configs/meetroom.json --batch_size 20000   --pretrained <pretrained_ckpt>  --n_iters 1000    --lr_sigma 0.3  --lr_sigma_final 0.3  --lr_sh 1e-2 --lr_sh_final 1e-4 --lr_sigma_decay_steps 1000 --lr_sh_decay_steps 1000   --frame_end 300 --fps 30 --train_use_all 0 --scale 1.0 --sh_keep_thres 1.0 --sh_prune_thres 0.1 --performance_mode  --dilate_rate_before 1 --dilate_rate_after 1 --stop_thres 0.01  --compress_saving --save_delta   --pilot_factor 2 
```

3. Train the full model

```bash
python train_video_n3dv_full.py -t <log_dir> <data_dir> -c configs/meetroom_full.json --batch_size 20000   --pretrained <pretrained_ckpt>  --n_iters 500    --lr_sigma 1.0 --lr_sigma_final 1.0  --lr_sh 1e-2 --lr_sh_final 1e-2 --lr_sigma_decay_steps 500 --lr_sh_decay_steps 500   --frame_end 300 --fps 30 --train_use_all 0 --scale 1.0 --sh_keep_thres 1.5 --sh_prune_thres 0.3 --performance_mode  --dilate_rate_before 2 --dilate_rate_after 2    --compress_saving --save_delta  --apply_narrow_band 
```

#### N3DV Dataset 

1. Initialize the first frame model

```bash
python opt.py  -t <log_dir> <data_dir>/0000 -c configs/init_ablation/n3dv_init.json --offset 500 --scale 0.5 --nosphereinit 
```

2. Train the pilot model
```bash
python train_video_n3dv_pilot.py -t <log_dir> <data_dir> -c configs/n3dv.json --batch_size 20000    --pretrained <pretrained_ckpt>  --n_iters 750    --lr_sigma 1.0  --lr_sigma_final 1.0  --lr_sh 1e-2 --lr_sh_final 1e-3 --lr_sigma_decay_steps 750 --lr_sh_decay_steps 750   --frame_end 300 --fps 30 --train_use_all 0 --offset 750  --scale 0.5 --sh_keep_thres 0.5 --sh_prune_thres 0.1 --performance_mode  --dilate_rate_before 1 --dilate_rate_after 1 --stop_thres 0.01  --compress_saving --save_delta   --pilot_factor 2 
```

3. Train the full model
```bash
python train_video_n3dv_full.py -t <log_dir> <data_dir> -c configs/n3dv_full.json --batch_size 20000   --pretrained  <pretrained_ckpt>  --n_iters 500    --lr_sigma 1.0 --lr_sigma_final 1.0  --lr_sh 1e-2 --lr_sh_final 3e-3 --lr_sigma_decay_steps 500 --lr_sh_decay_steps 300   --frame_end 300 --fps 30 --train_use_all 0 --offset 1500  --scale 0.5 --sh_keep_thres 1.0 --sh_prune_thres 0.2 --performance_mode  --dilate_rate_before 2 --dilate_rate_after 2  --stop_thres 0.01  --compress_saving --save_delta  --apply_narrow_band
```

## Testing StreamRF

For Meet Room Dataset：
```bash
python render_delta.py  -t <log_dir> <data_dir> -c configs/meetroom_full.json --batch_size 20000    --pretrained <pretrained_ckpt>  --frame_end 300 --fps 30 --scale 1.0 --performance_mode  
```

For N3DV Dataset：
```bash
python render_delta.py  -t <log_dir> <data_dir> -c configs/n3dv_full.json --batch_size 20000    --pretrained <pretrained_ckpt>  --frame_end 300 --fps 30 --scale 0.5 --performance_mode  
```
