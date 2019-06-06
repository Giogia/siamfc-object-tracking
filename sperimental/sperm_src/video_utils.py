import os
import numpy as np
import cv2


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


def run_video(queue_to_cnn, queue_to_video, frames):
    fps = 12
    p1 = 0, 0
    p2 = 0, 0

    queue_to_cnn.put(frames[0])
    queue_to_cnn.put(frames[1])
    queue_to_cnn.join()
    for frame in frames[1:]:

        if not queue_to_video.empty():
            queue_to_cnn.put(frame)
            bounding_box = queue_to_video.get()
            p1 = int(bounding_box[0]), int(bounding_box[1])
            p2 = int(bounding_box[0] + bounding_box[2]), int(bounding_box[1] + bounding_box[3])

        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow('Video', frame)
        cv2.waitKey(int(1000 / fps))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    queue_to_cnn.close()


p1_rect = [(0, 0)]
p2_rect = [(0, 0)]
flag_rect = 0
frame = 0
flag_cont = 0


def draw_rectangle(event, x, y, flag, param):
    global flag_rect, p1_rect, p2_rect, frame, flag_cont
    if event == cv2.EVENT_LBUTTONDOWN:
        p1_rect[0] = (x, y)
        flag_cont = 1

    if event == cv2.EVENT_LBUTTONUP:
        p2_rect[0] = (x, y)
        flag_rect = 1
        flag_cont = 0

    if event == cv2.EVENT_MOUSEMOVE and flag_cont == 1:
        cv2.rectangle(frame, p1_rect[0], (x, y), (0, 0, 255), 2)
        cv2.imshow('frame', frame)
        flag_no_sfliker = 1


def from_webcam(queue_to_cnn, queue_to_video):
    global flag_rect
    cap = cv2.VideoCapture(1)
    cv2.namedWindow('frame')
    cv2.setMouseCallback('frame', draw_rectangle)

    p1 = 0, 0
    p2 = 0, 0

    while True:
        # Capture frame-by-frame
        global frame
        ret, frame = cap.read()
        if flag_rect:
            cv2.rectangle(frame, p1_rect[0], p2_rect[0], (0, 0, 255), 2)
            width = p2_rect[0][0] - p1_rect[0][0]
            height = p2_rect[0][1] - p1_rect[0][1]
            initial_gt = (p1_rect[0][0] + width / 2, p1_rect[0][1] + height / 2, width, height)  # x_centre, y_centre,
            # w, h
            queue_to_cnn.put(initial_gt)
            queue_to_cnn.put(frame)
            queue_to_cnn.put(frame)
            queue_to_cnn.join()
            flag_rect = 0

        if not queue_to_video.empty():
            queue_to_cnn.put(frame)
            bounding_box = queue_to_video.get()
            p1 = int(bounding_box[0]), int(bounding_box[1])
            p2 = int(bounding_box[0] + bounding_box[2]), int(bounding_box[1] + bounding_box[3])

        cv2.rectangle(frame, p1, p2, (0, 0, 255), 2)
        cv2.imshow('frame', frame)
        cv2.waitKey(10)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    queue_to_cnn.close()
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


'''
Test

path = '../data'
videos = get_videos(path)

for video in videos:

    video_folder = os.path.join(path, video)
    frames = get_frames(video_folder)
    window = cv2.namedWindow("Video")
    run_video(frames, window)
    #gt = get_frames(video_folder)
    #video = initialize_video(video_folder)
    #print('\n\n', video)
    

'''
