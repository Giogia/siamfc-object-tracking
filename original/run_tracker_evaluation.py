import sys
import os
import numpy as np
from PIL import Image
import src.siamese as siam
from src.tracker import tracker
from src.parse_arguments import parse_arguments
from src.region_to_bbox import region_to_bbox


def main():
    # avoid printing TF debugging information
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # TODO: allow parameters from command line or leave everything in json files?
    hp, evaluation, run, env, design = parse_arguments()
    # Set size for use with tf.image.resize_images with align_corners=True.
    # For example,
    #   [1 4 7] =>   [1 2 3 4 5 6 7]    (length 3*(3-1)+1)
    # instead of
    # [1 4 7] => [1 1 2 3 4 5 6 7 7]  (length 3*3)
    final_score_sz = hp.response_up * (design.score_sz - 1) + 1
    # build TF graph once for all
    filename, image, templates_z, scores = siam.build_tracking_graph(final_score_sz, design, env)

    # iterate through all videos of evaluation.dataset
    if evaluation.video == 'all':
        dataset_folder = os.path.join(env.root_dataset, evaluation.dataset)
        videos_list = [v for v in os.listdir(dataset_folder)]
        videos_list.sort()
        nv = np.size(videos_list)
        speed = np.zeros(nv * evaluation.n_subseq)
        precisions = np.zeros(nv * evaluation.n_subseq)
        precisions_auc = np.zeros(nv * evaluation.n_subseq)
        ious = np.zeros(nv * evaluation.n_subseq)
        lengths = np.zeros(nv * evaluation.n_subseq)
        for i in range(nv):
            gt, frame_name_list, frame_sz, n_frames = _init_video(env, evaluation, videos_list[i])
            starts = np.rint(np.linspace(0, n_frames - 1, evaluation.n_subseq + 1))
            starts = starts[0:evaluation.n_subseq]
            for j in range(evaluation.n_subseq):
                start_frame = int(starts[j])
                gt_ = gt[start_frame:, :]
                frame_name_list_ = frame_name_list[start_frame:]
                pos_x, pos_y, target_w, target_h = region_to_bbox(gt_[0])
                idx = i * evaluation.n_subseq + j
                b_boxes, speed[idx] = tracker(hp, run, design, frame_name_list_, pos_x, pos_y,
                                              target_w, target_h, final_score_sz, filename,
                                              image, templates_z, scores, start_frame)
                lengths[idx], precisions[idx], precisions_auc[idx], ious[idx] = \
                    _compile_results(gt_, b_boxes, evaluation.dist_threshold)
                print('{} -- {} -- Precision: {.2f} -- Precisions AUC: {.2f} -- IOU: {.2f} -- Speed: {.2f} --'
                      .format(i, videos_list[i], precisions[idx], precisions_auc[idx], ious[idx], speed[idx]))

        tot_frames = np.sum(lengths)
        mean_precision = np.sum(precisions * lengths) / tot_frames
        mean_precision_auc = np.sum(precisions_auc * lengths) / tot_frames
        mean_iou = np.sum(ious * lengths) / tot_frames
        mean_speed = np.sum(speed * lengths) / tot_frames
        print('-- Overall stats (averaged per frame) on {} videos ({} frames) -- Precision ({} px): {} '
              '-- Precisions AUC: {} -- IOU: {} -- Speed: {} --'
              .format(nv, tot_frames, evaluation.dist_threshold, np.round(mean_precision), np.round(mean_precision_auc),
                      np.round(mean_iou), np.round(mean_speed)))

    else:
        gt, frame_name_list, _, _ = _init_video(env, evaluation, evaluation.video)
        pos_x, pos_y, target_w, target_h = region_to_bbox(gt[evaluation.start_frame])
        b_boxes, speed = tracker(hp, run, design, frame_name_list, pos_x, pos_y, target_w, target_h, final_score_sz,
                                 filename, image, templates_z, scores, evaluation.start_frame)
        _, precision, precision_auc, iou = _compile_results(gt, b_boxes, evaluation.dist_threshold)
        print('{} -- Precision ({} px): {} -- Precision AUC: {} -- IOU: {} -- Speed: {} --'
              .format(evaluation.video, evaluation.dist_threshold, np.round(precision, 2), np.round(precision_auc),
                      np.round(iou), np.round(speed)))


def _compile_results(gt, b_boxes, dist_threshold):
    l = np.size(b_boxes, 0)
    gt4 = np.zeros((l, 4))
    new_distances = np.zeros(l)
    new_ious = np.zeros(l)
    n_thresholds = 50
    precisions_ths = np.zeros(n_thresholds)

    for i in range(l):
        gt4[i, :] = region_to_bbox(gt[i, :], center=False)
        new_distances[i] = _compute_distance(b_boxes[i, :], gt4[i, :])
        new_ious[i] = _compute_iou(b_boxes[i, :], gt4[i, :])

    # what's the percentage of frame in which center displacement is inferior to given threshold? (OTB metric)
    precision = sum(new_distances < dist_threshold)/np.size(new_distances) * 100

    # find above result for many thresholds, then report the AUC
    thresholds = np.linspace(0, 25, n_thresholds+1)
    thresholds = thresholds[-n_thresholds:]
    # reverse it so that higher values of precision goes at the beginning
    thresholds = thresholds[::-1]
    for i in range(n_thresholds):
        precisions_ths[i] = sum(new_distances < thresholds[i])/np.size(new_distances)

    # integrate over the thresholds
    precision_auc = np.trapz(precisions_ths)

    # per frame averaged intersection over union (OTB metric)
    iou = np.mean(new_ious) * 100

    return l, precision, precision_auc, iou


def _init_video(env, evaluation, video):
    video_folder = os.path.join(env.root_dataset, evaluation.dataset, video)
    frame_name_list = [f for f in os.listdir(video_folder) if f.endswith(".jpg")]
    frame_name_list = [os.path.join(env.root_dataset, evaluation.dataset, video, '') + s for s in frame_name_list]
    frame_name_list.sort()
    with Image.open(frame_name_list[0]) as img:
        frame_sz = np.asarray(img.size)
        frame_sz[1], frame_sz[0] = frame_sz[0], frame_sz[1]

    # read the initialization from ground truth
    gt_file = os.path.join(video_folder, 'groundtruth.txt')
    gt = np.genfromtxt(gt_file, delimiter=',')
    n_frames = len(frame_name_list)
    assert n_frames == len(gt), 'Number of frames and number of GT lines should be equal.'

    return gt, frame_name_list, frame_sz, n_frames


def _compute_distance(box_a, box_b):
    a = np.array((box_a[0] + box_a[2] / 2, box_a[1] + box_a[3] / 2))
    b = np.array((box_b[0] + box_b[2] / 2, box_b[1] + box_b[3] / 2))
    dist = np.linalg.norm(a - b)

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
