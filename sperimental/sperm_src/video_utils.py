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

"""
def initialize_video(video_folder):
    gt = get_groundtruth(video_folder)
    frames_list = get_frames(video_folder)

    assert len(frames_list) == len(gt), 'Frames and groundtruth lines numbers should be equal.'

    return gt, frames_list
"""


def run_video(queue_to_cnn, queue_to_video, frames):
    fps = 12
    p1 = 0, 0
    p2 = 0, 0
    name_wind = 'Video'
    queue_to_cnn.put(frames[0])
    queue_to_cnn.put(frames[1])
    queue_to_cnn.join()
    for frame in frames[1:]:
        draw_bbox(int(1000 / fps), name_wind, frame, p1, p2, queue_to_cnn, queue_to_video)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    queue_to_cnn.close()


def draw_bbox(wait_time, name_wind, frame, p1, p2, queue_to_cnn, queue_to_video):
    if not queue_to_video.empty():
        queue_to_cnn.put(frame)
        bounding_box = queue_to_video.get()
        p1 = int(bounding_box[0]), int(bounding_box[1])
        p2 = int(bounding_box[0] + bounding_box[2]), int(bounding_box[1] + bounding_box[3])
        return p1, p2
    cv2.rectangle(frame, p1, p2, (255, 0, 0), 2)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow(name_wind, frame)
    cv2.waitKey(wait_time)


def draw_rectangle(event, x, y, flag, param):
    global flag_rect, drawing, points
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        points = ((x, y), (x, y))

    if event == cv2.EVENT_LBUTTONUP:
        points = (points[0], (x, y))
        drawing = False
        flag_rect = 1

    if event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            pass
            #frame_clean = frame.copy()


def from_webcam(queue_to_cnn, queue_to_video):
    global flag_rect, drawing
    flag_rect = 0
    drawing = False
    name_wind = 'Webcam'

    cap = cv2.VideoCapture(0)
    cv2.namedWindow(name_wind)
    cv2.setMouseCallback(name_wind, draw_rectangle)

    p1 = 0, 0
    p2 = 0, 0

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if flag_rect:

            cv2.rectangle(frame, points[0], points[1], (0, 0, 255), 2)
            width = points[1][0] - points[0][0]
            height = points[1][1] - points[0][1]
            initial_gt = (points[0][0] + width / 2, points[0][1] + height / 2, width, height)  # x_centre, y_centre,
            # w, h
            queue_to_cnn.put(initial_gt)
            queue_to_cnn.put(frame)
            queue_to_cnn.put(frame)
            queue_to_cnn.join()
            flag_rect = 0

        draw_bbox(10, name_wind, frame, p1, p2, queue_to_cnn, queue_to_video)

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
