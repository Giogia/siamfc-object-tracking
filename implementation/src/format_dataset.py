import os
import cv2
from shutil import copyfile
from src.video_utils import get_videos, get_frame_list

path = os.path.join('..', 'original', 'data', 'validation')
end_path = 'data'

os.mkdir(end_path)

videos = get_videos(path)

for video in videos:

    end_folder = os.path.join(end_path, video)
    os.mkdir(end_folder)

    video_folder = os.path.join(path, video)

    gt_folder = os.path.join(video_folder, 'groundtruth.txt')
    end_gt_path = os.path.join(end_path, video, 'groundtruth.txt')
    copyfile(gt_folder, end_gt_path)

    end_video_path = os.path.join(end_path, video, video + '.mp4')

    frames = get_frame_list(video_folder)
    height, width, channels = cv2.imread(frames[0]).shape
    writer = cv2.VideoWriter(end_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 24, (width, height))

    for frame in frames:
        writer.write(cv2.imread(frame))
    writer.release()
