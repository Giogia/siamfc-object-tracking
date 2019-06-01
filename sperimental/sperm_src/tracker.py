import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np
import time

import sperm_src.siamese_network as siamese_network
from sperm_src.visualization import show_frame
from sperm_src.parse import parameters


def tracker(frame_name_list, b_box_x, b_box_y, b_box_width, b_box_height, final_score_size, filename, image,
            network_z, scores):
    num_frames = np.size(frame_name_list)
    b_boxes = np.zeros((num_frames, 4))

    scale_factors = parameters.hyperparameters.scale_step ** np.linspace(
        -np.ceil(parameters.hyperparameters.scale_num / 2),
        np.ceil(parameters.hyperparameters.scale_num / 2),
        parameters.hyperparameters.scale_num)

    # Window to penalize large displacements
    hann_window = np.expand_dims(np.hanning(final_score_size), axis=0)

    penalty = np.transpose(hann_window) * hann_window
    penalty = penalty / np.sum(penalty)

    context = parameters.design.context * (b_box_width + b_box_height)
    window_size_z = np.sqrt((b_box_width + context) * (b_box_height + context))
    window_size_x = float(parameters.design.search_sz) / parameters.design.exemplar_sz * window_size_z

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        # Coordinate the loading of image files.
        coordinator = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coordinator)

        # Save first frame position (from ground-truth)
        b_boxes[0, :] = b_box_x - b_box_width / 2, b_box_y - b_box_height / 2, b_box_width, b_box_height

        image, network_z = sess.run([image, network_z], feed_dict={
            siamese_network.bbox_x_ph: b_box_x,
            siamese_network.bbox_y_ph: b_box_y,
            siamese_network.window_size_z_ph: window_size_z,
            filename: frame_name_list[0]})

        time_start = time.time()

        # Get an image from the queue
        for i in range(1, num_frames):
            scaled_window_size_z = window_size_z * scale_factors
            scaled_window_size_x = window_size_x * scale_factors
            scaled_b_box_width = b_box_width * scale_factors
            scaled_b_box_height = b_box_height * scale_factors
            print(frame_name_list[i])
            image, scores = sess.run(
                [image, scores],
                feed_dict={
                    siamese_network.bbox_x_ph: b_box_x,
                    siamese_network.bbox_y_ph: b_box_y,
                    siamese_network.window_size_x_0_ph: scaled_window_size_x[0],
                    siamese_network.window_size_x_1_ph: scaled_window_size_x[1],
                    siamese_network.window_size_x_2_ph: scaled_window_size_x[2],
                    network_z: np.squeeze(network_z),
                    filename: frame_name_list[i],
                })

            scores = np.squeeze(scores)
            # Penalize change of scale
            scores[0, :, :] = parameters.hyperparameters.scale_penalty * scores[0, :, :]
            scores[2, :, :] = parameters.hyperparameters.scale_penalty * scores[2, :, :]

            # Find scale with highest score
            best_scale = np.argmax(np.amax(scores, axis=(1, 2)))

            # Update scaled sizes
            window_size_x = (
                                    1 - parameters.hyperparameters.scale_lr) * window_size_x + parameters.hyperparameters.scale_lr * \
                            scaled_window_size_x[best_scale]
            b_box_width = (
                                  1 - parameters.hyperparameters.scale_lr) * b_box_width + parameters.hyperparameters.scale_lr * \
                          scaled_b_box_width[
                              best_scale]
            b_box_height = (
                                   1 - parameters.hyperparameters.scale_lr) * b_box_height + parameters.hyperparameters.scale_lr * \
                           scaled_b_box_height[
                               best_scale]

            # Select response with best scale
            best_score = scores[best_scale, :, :]
            best_score = best_score - np.min(best_score)
            best_score = best_score / np.sum(best_score)

            # Apply displacement penalty
            best_score = (
                                 1 - parameters.hyperparameters.window_influence) * best_score + parameters.hyperparameters.window_influence * penalty
            b_box_x, b_box_y = update_b_box_position(b_box_x, b_box_y, best_score, final_score_size, window_size_x)

            # Convert <cx,cy,w,h> to <x,y,w,h> and save output
            b_boxes[i, :] = b_box_x - b_box_width / 2, b_box_y - b_box_height / 2, b_box_width, b_box_height

            # Update the target representation with a rolling average
            if parameters.hyperparameters.z_lr > 0:
                new_network_z = sess.run([network_z], feed_dict={
                    siamese_network.bbox_x_ph: b_box_x,
                    siamese_network.bbox_y_ph: b_box_y,
                    siamese_network.window_size_z_ph: window_size_z,
                    image: image
                })

                network_z = (1 - parameters.hyperparameters.z_lr) * np.asarray(
                    network_z) + parameters.hyperparameters.z_lr * np.asarray(
                    new_network_z)

            # Update template patch size
            window_size_z = (
                                    1 - parameters.hyperparameters.scale_lr) * window_size_z + parameters.hyperparameters.scale_lr * \
                            scaled_window_size_z[best_scale]

            if parameters.run.visualization:
                show_frame(image, b_boxes[i, :], 1)

        time_elapsed = time.time() - time_start
        speed = num_frames / time_elapsed

        # Finish off the filename queue coordinator.
        coordinator.request_stop()
        coordinator.join(threads)

    plt.close('all')

    return b_boxes, speed


def update_b_box_position(b_box_x, b_box_y, score, final_score_size, window_size_x):
    search_sz = parameters.design.search_sz
    tot_stride = parameters.design.tot_stride
    response_up = parameters.hyperparameters.response_up

    # Find location of score maximizer
    point = np.array(np.unravel_index(np.argmax(score), np.shape(score)))

    # Displacement from the center in search area final representation
    center = float(final_score_size - 1) / 2
    disp_in_area = point - center

    # Displacement from the center in instance crop
    disp_in_window_x = disp_in_area * float(tot_stride) / response_up

    # Displacement from the center in instance crop (in frame coordinates)
    disp_in_frame = disp_in_window_x * window_size_x / search_sz

    # Position within frame in frame coordinates
    b_box_y, b_box_x = b_box_y + disp_in_frame[0], b_box_x + disp_in_frame[1]

    return b_box_x, b_box_y
