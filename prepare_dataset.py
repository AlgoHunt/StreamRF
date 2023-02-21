import os
import os.path as osp
import shutil
import argparse
import cv2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str)
    parser.add_argument('-n','--frames_num', type=int, default=300)
    parser.add_argument('-o','--output_dir', type=str, default=None)
    return parser.parse_args()

""" 
 Note: please install opencv first
 Extract every frame from video    
"""
def opencv_extractor(filename, outdir, prefix):
    cap = cv2.VideoCapture(filename)
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = 0
    while cap.isOpened():
        is_read, frame = cap.read()
        if not is_read:
            break
        cv2.imwrite(os.path.join(outdir, f'{count:04d}', 'images', prefix+'.png'), frame)
        count += 1
        

def prepare_data(data_dir, outdir, frame_num):
    files = os.listdir(data_dir)
    pose_bound = os.path.join(data_dir, 'poses_bounds.npy')
    assert os.path.exists(pose_bound), f'{pose_bound} file not found'
    print(f'step 1: copy [poses_bounds] to each directory')
    for i in range(frame_num):
        os.makedirs(os.path.join(outdir, f'{i:04d}'), exist_ok=True)
        shutil.copy(pose_bound, os.path.join(outdir, f'{i:04d}', 'poses_bounds.npy'))
        os.makedirs(os.path.join(outdir, f'{i:04d}', 'images'), exist_ok=True)

    print(f'step 2: extract frames from video')
    videos = [osp.join(data_dir, f) for f in files if f.endswith('mp4')]
    for video in videos:
        print(f'processing {video}')
        cam_num = int(video.split('/')[-1].split('.')[0].split('_')[1])
        opencv_extractor(video, outdir, prefix=f'cam{cam_num:02d}')
    
    print(f'Done!')



if __name__ == '__main__':
    args = parse_args()
    datadir = args.data_dir
    frameN = args.frames_num
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(datadir, 'frames')
    os.makedirs(output_dir, exist_ok=True)
    prepare_data(datadir, output_dir, frameN)
