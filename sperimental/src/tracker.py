import tensorflow as tf
import numpy as np
import src.siamese_network as siamese_network
from src.parse import parameters
from _queue import Empty


def tracker(queue_to_cnn, queue_to_video, final_score_size, region=None):
    image, network_z, input_scores = siamese_network.build_tracking_graph()

    if parameters.evaluation.video == 'webcam':
        region_to_bbox = queue_to_cnn.get()
        queue_to_cnn.task_done()
    else:
        region_to_bbox = region

    b_box_x, b_box_y, b_box_width, b_box_height = region_to_bbox

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

    initial_frame = queue_to_cnn.get()

    init = tf.global_variables_initializer()

    queue_to_cnn.task_done()

    with tf.Session() as sess:
        sess.run(init)

        # Save first frame position (from ground-truth)
        # b_boxes = (b_box_x - b_box_width / 2, b_box_y - b_box_height / 2, b_box_width, b_box_height)[np.newaxis, :]

        image_, network_z_ = sess.run([image, network_z], feed_dict={
            siamese_network.bbox_x_ph: b_box_x,
            siamese_network.bbox_y_ph: b_box_y,
            siamese_network.window_size_z_ph: window_size_z,
            siamese_network.frame: initial_frame})

        # Get an image from the queue
        while True:
            try:
                frame = queue_to_cnn.get(timeout=0.5)
            except Empty:
                break
            #scaled_window_size_z = window_size_z * scale_factors
            scaled_window_size_x = window_size_x * scale_factors
            scaled_b_box_width = b_box_width * scale_factors
            scaled_b_box_height = b_box_height * scale_factors
            image_, scores = sess.run(
                [image, input_scores],
                feed_dict={
                    siamese_network.bbox_x_ph: b_box_x,
                    siamese_network.bbox_y_ph: b_box_y,
                    siamese_network.window_size_x_0_ph: scaled_window_size_x[0],
                    siamese_network.window_size_x_1_ph: scaled_window_size_x[1],
                    siamese_network.window_size_x_2_ph: scaled_window_size_x[2],
                    network_z: np.squeeze(network_z_),
                    siamese_network.frame: frame,
                })

            scores = np.squeeze(scores)
            # Penalize change of scale
            scores[0, :, :] = parameters.hyperparameters.scale_penalty * scores[0, :, :]
            scores[2, :, :] = parameters.hyperparameters.scale_penalty * scores[2, :, :]

            # Find scale with highest score
            best_scale = np.argmax(np.amax(scores, axis=(1, 2)))

            # Update scaled sizes
            scale_lr = parameters.hyperparameters.scale_lr
            window_size_x = (1 - scale_lr) * window_size_x + scale_lr * scaled_window_size_x[best_scale]
            b_box_width = (1 - scale_lr) * b_box_width + scale_lr * scaled_b_box_width[best_scale]
            b_box_height = (1 - scale_lr) * b_box_height + scale_lr * scaled_b_box_height[best_scale]

            # Select response with best scale
            best_score = scores[best_scale, :, :]
            best_score = best_score - np.min(best_score)
            best_score = best_score / np.sum(best_score)

            # Apply displacement penalty
            window_influence = parameters.hyperparameters.window_influence
            best_score = (1 - window_influence) * best_score + window_influence * penalty
            b_box_x, b_box_y = update_b_box_position(b_box_x, b_box_y, best_score, final_score_size, window_size_x)

            b_box = b_box_x - b_box_width / 2, b_box_y - b_box_height / 2, b_box_width, b_box_height
            #np.append(b_boxes, b_box, axis=0)
            queue_to_video.put(b_box)
            queue_to_cnn.task_done()
            """
            # Convert <cx,cy,w,h> to <x,y,w,h> and save output
            # b_boxes = b_box_x - b_box_width / 2, b_box_y - b_box_height / 2, b_box_width, b_box_height

            # Update the target representation with a rolling average
            if parameters.hyperparameters.z_lr > 0:
                new_network_z = sess.run([network_z], feed_dict={
                    siamese_network.bbox_x_ph: b_box_x,
                    siamese_network.bbox_y_ph: b_box_y,
                    siamese_network.window_size_z_ph: window_size_z,
                    image: image_
                })

                z_lr = parameters.hyperparameters.z_lr
                network_z_ = (1 - z_lr) * np.asarray(network_z_) + z_lr * np.asarray(new_network_z)

            # Update template patch size
            window_size_z = (1 - scale_lr) * window_size_z + scale_lr * scaled_window_size_z[best_scale]
            """
    #return b_boxes


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

'''
def show_frame(image, bounding_box, window):
    while True:
        #image = image.astype(dtype=np.uint8)
        #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
        p1 = tuple(bounding_box[:2].astype(np.int))
        p2 = tuple((bounding_box[:2]+bounding_box[2:]).astype(np.int))
        cv2.rectangle(image, p1, p2, (0, 0, 255), 2)
        cv2.imshow(window, image)
        cv2.waitKey(1)
        break
'''