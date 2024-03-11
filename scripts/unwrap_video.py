#%%
# # script to unwrap video of rotating object into a panorama
# example usage: "python unwrap_video.py -p path/to/video -s path/to/save"
# make sure to place "TileConfiguration.txt" in the same folder as the videos of interest

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
import scyjava

# default parameter
fiji_path = '~/home/shared/Fiji-installation/Fiji.app'

# pulling user-input variables from command line
parser = argparse.ArgumentParser(description='script to unwrap a video into a panorama')
parser.add_argument('-p', '--video-path', dest='video_path', action='store', type=str, required=True, help='location of video file')
parser.add_argument('-s', '--save-path', dest='save_path', action='store', type=str, required=True, help='the folder in which to save unwrapped file')
parser.add_argument('-f', '--fiji-path', dest='fiji_path', action='store', type=str, default=fiji_path, help='point to local installation of Fiji')
parser.add_argument('-t', '--tile-config', dest='tile_config', action='store', type=str, default=True, help='use tile configuration file')

# ingesting user-input arguments
args = parser.parse_args()
video_path = args.video_path
save_path = args.save_path
fiji_path = args.fiji_path
tile_config = args.tile_config

# extract frames from video and crop centre 150 pixels
def extract_frames(video_path, interval=1, save_path='', crop=[525, 675], stop_frame = 250): #250
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
        if count <= stop_frame and (count % interval == 0):  # Save frame every 'interval' frames
            frames.append(image)
        success, image = vidcap.read()
        count += 1

    vidcap.release()
 
    os.makedirs(f'{save_path}/sequence/', exist_ok=True)
    for i, frame in enumerate(frames):
        frame = frame[:, crop[0]:crop[1]]
        cv2.imwrite(f'{save_path}/sequence/{str(i).zfill(3)}.jpg', frame)

    return(frames)

def stitch_images(frames, save_path, tile_config, name):
    path = f'{save_path}/sequence'
    if(tile_config==True):
        source_path = f'{save_path}/TileConfiguration.txt'
        destination_path = f'{path}/TileConfiguration.txt'
        shutil.copy2(source_path, destination_path)

    print(f'frame count {len(frames)}')

    plugin = "Grid/Collection stitching"
    args = {
        "type": "[Grid: row-by-row]",
        "order": "[Right & Down]",
        "grid_size_x": f"{(len(frames))}",
        "grid_size_y": "1",
        "tile_overlap": "86",
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

    # if tile_config=True, stitch based on tile configuration file
    if(tile_config==True):
        args = {
            "type": "[Positions from file]",
            "order": "[Defined by TileConfiguration]",
            "layout_file": f"TileConfiguration.txt",
            "directory": f'{path}',
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

    # Define crop box with left, upper, right, and lower coordinates; heuristically defined by looking at uncropped images
    crop_box = (0, 0, 1045, image_rgb.height)  # Right coordinate is 1050, lower coordinate is the height of the image

    # Crop the image
    cropped_image_rgb = image_rgb.crop(crop_box)

    # save the image
    cropped_image_rgb.save(f'{save_path}/UNWRAPPED_{name}.jpg')
    
    # delete everything from sequence directory and then directory itself
    try:
        shutil.rmtree(f'{path}/')
    except:
        print('Cannot delete folder!')
    

# Start ImageJ
scyjava.config.add_option('-Xmx6g')
ij = imagej.init(fiji_path) # point to local installation
#ij = imagej.init('sc.fiji:fiji')
#ij.getVersion()

# process single video
if(os.path.isfile(video_path)):
    frames = extract_frames(video_path, interval=5, save_path=save_path)
    name = os.path.basename(video_path)
    stitch_images(frames, save_path, tile_config, name)

# batch process videos in folder
if(os.path.isdir(video_path)):
    video_files = [f'{video_path}/{f}' for f in os.listdir(video_path) if os.path.isfile(os.path.join(video_path, f)) and not (f.endswith('.txt') or f=='.DS_Store')]
    for video_file_path in video_files:
        frames = extract_frames(video_file_path, interval=5, save_path=save_path)
        name = os.path.basename(video_file_path)
        stitch_images(frames, save_path, tile_config, name)
