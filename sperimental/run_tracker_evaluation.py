import sys
import os
import numpy as np
import cv2
import multiprocessing as mp
import sperm_src.siamese_network as siam
from sperm_src.parse import parameters
from sperm_src.bounding_box import region_to_bbox
from sperm_src.tracker import tracker
from sperm_src.video_utils import *


def main():

    evaluation, environment, hyperparameters, design = \
        parameters.evaluation, parameters.environment, parameters.hyperparameters, parameters.design
    # avoid printing TF debugging information
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    final_score_sz = hyperparameters.response_up * (design.score_sz - 1) + 1
    # build TF graph once for all


    # iterate through all videos of evaluation.dataset
    if evaluation.video == 'all':
        videos_list = sorted([v for v in os.listdir(environment.dataset_folder)])
    else:
        videos_list = [evaluation.video]

    nv = np.size(videos_list)
    precisions, precisions_auc, ious, lengths = \
        np.zeros(nv), np.zeros(nv), np.zeros(nv), np.zeros(nv)

    for i in range(nv):

        video_path = os.path.join(environment.dataset_folder, videos_list[i])

        queue_to_video = mp.Queue(maxsize=1)
        queue_to_cnn = mp.JoinableQueue(maxsize=2)

        frames = get_frames(video_path)
        gt = get_groundtruth(video_path)
        region = region_to_bbox(gt[0])

        process_tracker = mp.Process(target=tracker, args=(queue_to_cnn, queue_to_video, region, final_score_sz))
        process_video = mp.Process(target=run_video, args=(queue_to_cnn, queue_to_video, frames))

        process_video.start()
        process_tracker.start()

        '''
        b_boxes = tracker(queue_to_cnn, queue_to_video, finish_value, region,
                          final_score_sz, image, templates_z, scores)

        lengths[i], precisions[i], precisions_auc[i], ious[i] = \
            _compile_results(gt, b_boxes, evaluation.dist_threshold)
        if evaluation.video == 'all':
            print('{} -- {} -- Precision: {} -- Precisions AUC: {} -- IOU: {} --'
                  .format(i, videos_list[i], np.round(precisions[i], 2),
                          np.round(precisions_auc[i], 2), np.round(ious[i], 2)))
        '''

        process_video.join()
        process_tracker.join()

    '''
    tot_frames = np.sum(lengths)
    mean_precision = np.sum(precisions * lengths) / tot_frames
    mean_precision_auc = np.sum(precisions_auc * lengths) / tot_frames
    mean_iou = np.sum(ious * lengths) / tot_frames
    print('-- Overall stats (averaged per frame) on {} videos ({} frames) -- Precision ({} px): {} '
          '-- Precisions AUC: {} -- IOU: {} --'
          .format(nv, tot_frames, evaluation.dist_threshold, np.round(mean_precision),
                  np.round(mean_precision_auc), np.round(mean_iou)))
                  
    '''


def _compile_results(gt, b_boxes, dist_threshold):
    length = np.size(b_boxes, 0)
    gt4 = np.zeros((length, 4))
    new_distances = np.zeros(length)
    new_ious = np.zeros(length)
    n_thresholds = 50

    for i in range(length):
        gt4[i, :] = region_to_bbox(gt[i, :], center=False)  # gt coordinates are already centered
        new_distances[i] = _compute_distance(b_boxes[i, :], gt4[i, :])
        new_ious[i] = _compute_iou(b_boxes[i, :], gt4[i, :])

    # what's the percentage of frame in which center displacement is inferior to given threshold? (OTB metric)
    precision = sum(new_distances < dist_threshold)/np.size(new_distances) * 100

    # find above result for many thresholds, then report the AUC
    thresholds = np.linspace(0, 25, n_thresholds+1)[1:]
    # reverse it so that higher values of precision goes at the beginning
    thresholds = thresholds[::-1]
    precisions_ths = [sum(new_distances < thresholds[i])/np.size(new_distances) for i in range(n_thresholds)]

    # integrate over the thresholds
    precision_auc = np.trapz(precisions_ths)

    # per frame averaged intersection over union (OTB metric)
    iou = np.mean(new_ious) * 100

    return length, precision, precision_auc, iou


def _compute_distance(box_a, box_b):  # x, y, w, h
    a = np.array((box_a[0] + box_a[2] / 2, box_a[1] + box_a[3] / 2))
    b = np.array((box_b[0] + box_b[2] / 2, box_b[1] + box_b[3] / 2))
    dist = np.linalg.norm(a - b)  # distance between centers of the bounding boxes

    assert dist >= 0
    assert dist != float('Inf')

    return dist


def _compute_iou(box_a, box_b):
    # determine the (x, y)-coordinates of the intersection rectangle
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[0] + box_a[2], box_b[0] + box_b[2])
    y_b = min(box_a[1] + box_a[3], box_b[1] + box_b[3])

    if x_a < x_b and y_a < y_b:
        # compute the area of intersection rectangle
        inter_area = (x_b - x_a) * (y_b - y_a)
        # compute the area of both the prediction and ground-truth
        # rectangles
        box_a_area = box_a[2] * box_a[3]
        box_b_area = box_b[2] * box_b[3]
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the intersection area
        iou = inter_area / float(box_a_area + box_b_area - inter_area)
    else:
        iou = 0

    assert iou >= 0
    assert iou <= 1.01

    return iou


if __name__ == '__main__':
    sys.exit(main())
