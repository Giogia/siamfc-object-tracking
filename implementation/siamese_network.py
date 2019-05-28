import tensorflow as tf
import os.path
import numpy as np

from crop_frames import *

bbox_x_ph = tf.placeholder(tf.float64, name="bbox_x")
bbox_y_ph = tf.placeholder(tf.float64, name="bbox_y")

window_size_z_ph = tf.placeholder(tf.float64, name="window_size")
window_size_x_0_ph = tf.placeholder(tf.float64)
window_size_x_1_ph = tf.placeholder(tf.float64)
window_size_x_2_ph = tf.placeholder(tf.float64)


# the follow parameters *have to* reflect the design of the network to be imported
_conv_stride = np.array([2, 1, 1, 1, 1])
_filtergroup_yn = np.array([0, 1, 0, 1, 1], dtype=bool)
_bnorm_yn = np.array([1, 1, 1, 1, 0], dtype=bool)
_relu_yn = np.array([1, 1, 1, 1, 0], dtype=bool)
_pool_stride = np.array([2, 1, 0, 0, 0])  # 0 means no pool
_pool_sz = 3
_bnorm_adjust = True
assert len(_conv_stride) == len(_filtergroup_yn) == len(_bnorm_yn) == len(_relu_yn) == len(_pool_stride), (
    'These arrays of flags should have same length')
assert all(_conv_stride) >= True, 'The number of conv layers is assumed to define the depth of the network'
_num_layers = len(_conv_stride)


def build_tracking_graph():

    filename = tf.placeholder(tf.string, [], name='filename')

    # Turn image into a Tensor
    image = tf.read_file(filename)
    image = tf.image.decode_jpeg(image)
    image = 255.0 * tf.image.convert_image_dtype(image, tf.float32)

    # Pad frames if necessary
    padded_frame_z, padding_z = frame_padding(image, bbox_x_ph, bbox_y_ph, window_size_z_ph)
    padded_frame_z = tf.cast(padded_frame_z, tf.float32)

    padded_frame_x, padding_x = frame_padding(image, bbox_x_ph, bbox_y_ph, window_size_x_2_ph)
    padded_frame_x = tf.cast(padded_frame_x, tf.float32)

    # Extract tensor Z
    tensor_z = crop_resize(padded_frame_z, padding_z, bbox_x_ph, bbox_y_ph, [window_size_z_ph],
                          parameters.design.exemplar_sz)

    # Extract tensor X (3 scales)
    tensor_x = crop_resize(padded_frame_x, padding_x, bbox_x_ph, bbox_y_ph,
                           [window_size_x_0_ph, window_size_x_1_ph, window_size_x_2_ph],
                           parameters.design.search_sz)


    # TODO FINISH CREATE SIAMESE AND DECOMMENT
    '''
    # Create MatConvnet pre-trained fully-convolutional Siamese network
    template_z, templates_x, p_names_list, p_val_list = _create_siamese(os.path.join(env.root_pretrained, design.net),
                                                                        tensor_z, tensor_x)
    template_z = tf.squeeze(template_z)
    templates_z = tf.stack([template_z, template_z, template_z])
    # compare templates via cross-correlation
    scores = _match_templates(templates_z, templates_x, p_names_list, p_val_list)
    # upsample the score maps
    scores_up = tf.image.resize_images(scores, [final_score_sz, final_score_sz],
                                       method=tf.image.ResizeMethod.BICUBIC, align_corners=True)
    return filename, image, templates_z, scores_up
    '''


    