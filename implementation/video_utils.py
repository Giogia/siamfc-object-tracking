import os
import numpy as np
from PIL import Image


def get_videos(dataset_folder):

    videos_list = [video for video in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder, video))]
    videos_list.sort()

    return videos_list


def get_frames(video_folder):

    frames_list = [frame for frame in os.listdir(video_folder) if frame.endswith('.jpg')]
    # frames_list = [os.path.join(parameters.environment.dataset_folder, video, '') + s for s in frames_list]
    frames_list = [os.path.join(video_folder, s) for s in frames_list]
    frames_list.sort()

    return frames_list


def get_groundtruth(video_folder):

    gt_file = os.path.join(video_folder, 'groundtruth.txt')
    gt = np.genfromtxt(gt_file, delimiter=',')

    return gt


def initialize_video(video_folder):

    gt = get_groundtruth(video_folder)
    frames_list = get_frames(video_folder)

    with Image.open(frames_list[0]) as img:
        frame_size = np.flip(np.asarray(img.size))

    frames_number = len(frames_list)
    assert frames_number == len(gt), 'Frames and groundtruth lines numbers should be equal.'

    return gt, frames_list, frame_size, frames_number


'''
Test

path = '../original/data/validation'
videos = get_videos(path)

for video in videos:

    video = os.path.join(path, video)
    frames = get_frames(video)
    gt = get_groundtruth(video)

    video = initialize_video(video)
    print('\n\n', video)

'''
