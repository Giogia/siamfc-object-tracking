import os
import time

import cv2
import numpy as np

from src.parse_arguments import parameters


def get_videos(dataset_folder):
    videos_list = [video for video in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder, video))]
    videos_list.sort()

    return videos_list


def get_frame_list(video_folder):
    frames_list = [frame for frame in os.listdir(video_folder) if frame.endswith('.jpg')]
    frames_list = [os.path.join(video_folder, s) for s in frames_list]
    frames_list.sort()

    return frames_list


def get_frames(video_folder):
    frames_list = []
    video_capture = cv2.VideoCapture(os.path.join(video_folder, os.path.basename(video_folder) + '.mp4'))
    success, frame = video_capture.read()
    while success:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames_list.append(frame)
        success, frame = video_capture.read()

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


def show_frame(image, bounding_box):
    while True:
        image = image.astype(dtype=np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = draw_frame(image, bounding_box)
        cv2.imshow("", image)
        cv2.waitKey(1)
        break


def draw_frame(image, bounding_box):
    p1 = tuple(bounding_box[:2].astype(np.int))
    p2 = tuple((bounding_box[:2] + bounding_box[2:]).astype(np.int))
    cv2.rectangle(image, p1, p2, (0, 0, 255), 2)

    return image


def save_video(frame_list, b_boxes):

    assert len(frame_list) == len(b_boxes), 'The number of frames doesnt correspond to number of bounding boxes'

    path = os.path.join(parameters.environment.results_folder, str(time.time()) + '.mp4')
    frame_list = [frame.astype(dtype=np.uint8) for frame in frame_list]
    frame_list = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frame_list]
    frames = [draw_frame(frame, b_box) for frame, b_box in zip(frame_list, b_boxes)]
    height, width, channels = frames[0].shape
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), 24, (width, height))

    for frame in frames:
        writer.write(frame)

    writer.release()

'''
Test

path = '../data'
videos = get_videos(path)

for video in videos:

    video_folder = os.path.join(path, video)
    frames = get_frames(video_folder)
    gt = get_frames(video_folder)
    video = initialize_video(video_folder)
    print(len(frames))
    #print('\n\n', video)

'''