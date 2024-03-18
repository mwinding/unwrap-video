#!/bin/bash
# usage explanation:
# sbatch --export=VIDEO_PATH="/path/to/video_folder,SAVE_PATH="path/to/save_folder",TILE_CONFIG=True sbatch-unwrap.sh

#SBATCH --job-name=unwrap_videos
#SBATCH --ntasks=1
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-%j.out
#SBATCH --mail-user=$(whoami)@crick.ac.uk
#SBATCH --mail-type=FAIL

ml purge
ml Anaconda3/2023.09-0
source /camp/apps/eb/software/Anaconda/conda.env.sh

# unwrap videos
conda activate pyimagej-env
python unwrap_video.py -f /camp/home/shared/Fiji-installation/Fiji.app -p $VIDEO_PATH -s $SAVE_PATH -t $TILE_CONFIG


