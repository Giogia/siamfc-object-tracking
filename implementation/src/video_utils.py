import os
import numpy as np
import cv2
from PIL import Image


def get_videos(dataset_folder):

    videos_list = [video for video in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder, video))]
    videos_list.sort()

    return videos_list


def get_frames(video_folder):

    frames_list = []
    video_capture = cv2.VideoCapture(os.path.join(video_folder, os.path.basename(video_folder) + '.mp4'))
    success, frame = video_capture.read()
    while success:
        success, frame = video_capture.read()
        frames_list.append(frame)

    return frames_list


def get_groundtruth(video_folder):

    gt_file = os.path.join(video_folder, 'groundtruth.txt')
    gt = np.genfromtxt(gt_file, delimiter=',')

    return gt


def initialize_video(video_folder):

    gt = get_groundtruth(video_folder)
    frames_list = get_frames(video_folder)

    assert len(frames_list) == len(gt), 'Frames and groundtruth lines numbers should be equal.'

    return gt, frames_list


'''
Test

path = 'data'
videos = get_videos(path)

for video in videos:

    video_folder = os.path.join(path, video)
    frames = get_frames(video_folder)
    gt = get_frames(video_folder)
    video = initialize_video(video_folder)
    print('\n\n', video)
'''

