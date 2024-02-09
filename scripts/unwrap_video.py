# script to unwrap video of rotating object into a panorama
# example usage: "python unwrap_video.py path/to/video path/to/save"

# uses Fiji!
# requires a bit of installation, 
# 1. create fiji environment via "conda create -n pyimagej-env openjdk=11 python=3.09"
# 2. navigate to environment "conda activate pyimagej-env"
# 3. "pip install pyimagej"
# [4. install maven "brew install maven"]

import cv2
import numpy as np
import pandas as pd
import os
import argparse
import imagej
import numpy as np
from PIL import Image
import shutil

# default parameter
fiji_path = 'home/shared/Fiji-installation/Fiji.app'

# pulling user-input variables from command line
parser = argparse.ArgumentParser(description='script to unwrap a video into a panorama')
parser.add_argument('-p', '--video-path', dest='video_path', action='store', type=str, required=True, help='location of video file')
parser.add_argument('-s', '--save-path', dest='save_path', action='store', type=str, required=True, help='the folder in which to save unwrapped file')
parser.add_argument('-f', '--fiji-path', dest='save_path', action='store', type=str, default=fiji_path, help='point to local installation of Fiji')

# ingesting user-input arguments
args = parser.parse_args()
video_path = args.video_path
save_path = args.save_path
fiji_path = args.fiji_path

# extract frames from video
def extract_frames(video_path, interval=1, save=False, save_path=''):
    """
    Extract frames from a video.
    
    :param video_path: Path to the video file.
    :param interval: Interval of frames to extract (1 = every frame, 2 = every other frame, etc.)
    """
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    frames = []

    while success:
        if count % interval == 0:  # Save frame every 'interval' frames
            frames.append(image)
        success, image = vidcap.read()
        count += 1

    vidcap.release()
    # 
    if(save):
        os.makedirs(f'{save_path}/sequence/')
        for i, frame in enumerate(frames):
            cv2.imwrite(f'{save_path}/sequence/{str(i).zfill(3)}.jpg', frame)

    return(frames)

frames = extract_frames(video_path, interval=3, save=True, save_path=save_path)

# crop video appropriately
## add later ##

# Start ImageJ
ij = imagej.init(fiji_path) # point to local installation
#ij = imagej.init('sc.fiji:fiji')
#ij.getVersion()

path = f'{save_path}/sequence'

plugin = "Grid/Collection stitching"
args = {
    "type": "[Grid: row-by-row]",
    "order": "[Right & Down]",
    "grid_size_x": f"{len(frames)}",
    "grid_size_y": "1",
    "tile_overlap": "95",
    "first_file_index_i": "0",
    "directory": f'{path}',
    "file_names": "{iii}.jpg",
    "output_textfile_name": "TileConfiguration.txt",
    "fusion_method": "[Linear Blending]",
    "regression_threshold": "0.30",
    "max/avg_displacement_threshold": "2.50",
    "absolute_displacement_threshold": "3.50",
    "compute_overlap": True,
    "computation_parameters": "[Save computation time (but use more RAM)]",
    "image_output": "[Write to disk]",
    "output_directory": f'{path}'
}

# run plugin
ij.py.run_plugin(plugin, args)

# Fiji stitcher saves output as separate 8-bit R, G, and B images
# merge them together and save here

# Open the 8-bit grayscale TIFF images
image_r = Image.open(f'{path}/img_t1_z1_c1')
image_g = Image.open(f'{path}/img_t1_z1_c2')
image_b = Image.open(f'{path}/img_t1_z1_c3')

# Merge the images into one RGB image
image_rgb = Image.merge('RGB', (image_r, image_g, image_b))

# save the image
image_rgb.save(f'{save_path}/stitched-image.jpg')

# delete everything from sequence directory and then directory itself
try:
    shutil.rmtree(f'{path}/')
except:
    print('Cannot delete folder!')
