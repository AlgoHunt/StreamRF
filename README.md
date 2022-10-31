# Streaming Radiance Fields for 3D Video Synthesis

Lingzhi Li, Zhen Shen, Zhongshu Wang, Li Shen, Ping Tan

Alibaba Group


@inproceedings{li2022streamrf,
      title={Streaming Radiance Fields for 3D Video Synthesis}, 
      author={Lingzhi Li and Zhen Shen and Zhongshu Wang and Li Shen and Ping Tan},
      year={2022},
      booktitle={NeurIPS},
}

arXiv: <https://arxiv.org/abs/2210.14831>

**Note:**  We will add detailed illustration in the near future.


## Dataset
**Meet Room Dataset**: coming soon


**N3DV Dataset**:
https://github.com/facebookresearch/Neural_3D_Video

## Training StreamRF


### Meet Room Dataset

1. Train the pilot model

```
python train_video_n3dv_pilot.py -t <log_dir> <data_dir> -c configs/meetroom.json --batch_size 20000   --pretrained <pretrained_ckpt>  --n_iters 1000    --lr_sigma 0.3  --lr_sigma_final 0.3  --lr_sh 1e-2 --lr_sh_final 1e-4 --lr_sigma_decay_steps 1000 --lr_sh_decay_steps 1000   --frame_end 300 --fps 30 --train_use_all 0 --scale 1.0 --sh_keep_thres 1.0 --sh_prune_thres 0.1 --performance_mode  --dilate_rate_before 1 --dilate_rate_after 1 --stop_thres 0.01  --compress_saving --save_delta   --pilot_factor 2 
```

2. Train the full model

```
python train_video_n3dv_full.py -t <log_dir> <data_dir> -c configs/meetroom_full.json --batch_size 20000   --pretrained <pretrained_ckpt>  --n_iters 500    --lr_sigma 1.0 --lr_sigma_final 1.0  --lr_sh 1e-2 --lr_sh_final 1e-2 --lr_sigma_decay_steps 500 --lr_sh_decay_steps 500   --frame_end 300 --fps 30 --train_use_all 0 --scale 1.0 --sh_keep_thres 1.5 --sh_prune_thres 0.3 --performance_mode  --dilate_rate_before 2 --dilate_rate_after 2    --compress_saving --save_delta  --apply_narrow_band 
```

#### N3DV Dataset 

1. Train the pilot model
```bash
python train_video_n3dv_pilot.py -t <log_dir> <data_dir> -c configs/n3dv.json --batch_size 20000    --pretrained <pretrained_ckpt>  --n_iters 750    --lr_sigma 1.0  --lr_sigma_final 1.0  --lr_sh 1e-2 --lr_sh_final 1e-3 --lr_sigma_decay_steps 750 --lr_sh_decay_steps 750   --frame_end 300 --fps 30 --train_use_all 0 --offset 750  --scale 0.5 --sh_keep_thres 0.5 --sh_prune_thres 0.1 --performance_mode  --dilate_rate_before 1 --dilate_rate_after 1 --stop_thres 0.01  --compress_saving --save_delta   --pilot_factor 2 
```

2. Train the full model
```bash
python train_video_n3dv_full.py -t <log_dir> <data_dir> -c configs/n3dv_full.json --batch_size 20000   --pretrained  <pretrained_ckpt>  --n_iters 500    --lr_sigma 1.0 --lr_sigma_final 1.0  --lr_sh 1e-2 --lr_sh_final 3e-3 --lr_sigma_decay_steps 500 --lr_sh_decay_steps 300   --frame_end 300 --fps 30 --train_use_all 0 --offset 1500  --scale 0.5 --sh_keep_thres 1.0 --sh_prune_thres 0.2 --performance_mode  --dilate_rate_before 2 --dilate_rate_after 2  --stop_thres 0.01  --compress_saving --save_delta  --apply_narrow_band
```

## Testing StreamRF
coming soon
