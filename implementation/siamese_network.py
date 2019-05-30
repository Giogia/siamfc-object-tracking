import tensorflow as tf
import os.path
import numpy as np

from crop_frames import *

# Parameters reflecting the design of the network
CONV_STRIDE = np.array([2, 1, 1, 1, 1])
FILTERGROUP = np.array([0, 1, 0, 1, 1], dtype=bool)
BNORM = np.array([1, 1, 1, 1, 0], dtype=bool)
RELU = np.array([1, 1, 1, 1, 0], dtype=bool)
POOL_STRIDE = np.array([2, 1, 0, 0, 0])  # 0 means no pool
POOL_SIZE = 3
BNORM_ADJUST = True

NUM_LAYERS = len(CONV_STRIDE)

# Check network structure
assert len(CONV_STRIDE) == len(FILTERGROUP) == len(BNORM) == len(RELU) == len(POOL_STRIDE), (
    'These arrays of flags should have same length')
assert all(CONV_STRIDE) >= True, 'The number of conv layers is assumed to define the depth of the network'

# Network Placeholders
bbox_x_ph = tf.placeholder(tf.float64, name="bbox_x")
bbox_y_ph = tf.placeholder(tf.float64, name="bbox_y")

window_size_z_ph = tf.placeholder(tf.float64, name="window_size")
window_size_x_0_ph = tf.placeholder(tf.float64)
window_size_x_1_ph = tf.placeholder(tf.float64)
window_size_x_2_ph = tf.placeholder(tf.float64)


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

    template_z, templates_x, p_names_list, p_val_list = create_network(tensor_x, tensor_z)


    # TODO FINISH CREATE SIAMESE AND DECOMMENT
    '''
    template_z = tf.squeeze(template_z)
    templates_z = tf.stack([template_z, template_z, template_z])
    # compare templates via cross-correlation
    scores = _match_templates(templates_z, templates_x, p_names_list, p_val_list)
    # upsample the score maps
    scores_up = tf.image.resize_images(scores, [final_score_sz, final_score_sz],
                                       method=tf.image.ResizeMethod.BICUBIC, align_corners=True)
    return filename, image, templates_z, scores_up
    '''


def create_network(tensor_x, tensor_z):

    model_path = os.path.join(parameters.environment.model_folder, parameters.environment.model_name)

    for i in range(NUM_LAYERS):

        print('> Layer ' + str(i + 1))

        conv_w_name = 'conv' + str(i + 1) + '/W'
        conv_b_name = 'conv' + str(i + 1) + '/b'

        print('\t\tCONV: setting ' + conv_w_name + ' ' + conv_b_name)
        print('\t\tCONV: stride ' + str(CONV_STRIDE[i]) + ', filter-group ' + str(FILTERGROUP[i]))

        conv_w = tf.train.load_variable(ckpt_dir_or_file=model_path, name=conv_w_name)
        conv_b = tf.train.load_variable(ckpt_dir_or_file=model_path, name=conv_b_name)

        if BNORM[i]:
            bn_beta_name = 'conv' + str(i + 1) + '/batch_normalization/beta'
            bn_gamma_name = 'conv' + str(i + 1) + '/batch_normalization/gamma'
            bn_moving_mean_name = 'conv' + str(i + 1) + '/batch_normalization/moving_mean'
            bn_moving_variance_name = 'conv' + str(i + 1) + '/batch_normalization/moving_variance'

            print('\t\tBNORM: setting ' + bn_beta_name + ' ' + bn_gamma_name + ' ' + bn_moving_mean_name + ' ' + bn_moving_variance_name)

            bn_beta = tf.train.load_variable(ckpt_dir_or_file=model_path, name=bn_beta_name)
            bn_gamma = tf.train.load_variable(ckpt_dir_or_file=model_path, name=bn_gamma_name)
            bn_moving_mean = tf.train.load_variable(ckpt_dir_or_file=model_path, name=bn_moving_mean_name)
            bn_moving_variance = tf.train.load_variable(ckpt_dir_or_file=model_path, name=bn_moving_variance_name)

        else:
            bn_beta = bn_gamma = bn_moving_mean = bn_moving_variance = []

        # set up conv "block" with bnorm and activation
        tensor_x = set_convolutional(tensor_x, conv_w, np.swapaxes(conv_b, 0, 1), _conv_stride[i],
                                     bn_beta, bn_gamma, bn_moving_mean, bn_moving_variance,
                                     filter_group=_filtergroup_yn[i], batch_norm=_bnorm_yn[i], activation=_relu_yn[i],
                                     scope='conv' + str(i + 1), reuse=False)

        # notice reuse=True for Siamese parameters sharing
        tensor_z = set_convolutional(tensor_z, conv_w, np.swapaxes(conv_b, 0, 1), _conv_stride[i], bn_beta, bn_gamma,
                                     bn_moving_mean, bn_moving_variance, filter_group=_filtergroup_yn[i],
                                     batch_norm=_bnorm_yn[i], activation=_relu_yn[i], scope='conv' + str(i + 1),
                                     reuse=True)

        return net_z, net_x, params_names_list, params_values_list


def set_convolutional(tensor, window, b, stride, batch_norm_beta, batch_norm_gamma, batch_norm_mm, batch_norm_mv, filter_group=False, batch_norm=True,
                      activation=True, scope=None, reuse=False):
    # use the input scope or default to "conv"
    with tf.variable_scope(scope or 'conv', reuse=reuse):
        # sanity check
        window = tf.get_variable("W", window.shape, trainable=False, initializer=tf.constant_initializer(window))
        b = tf.get_variable("b", b.shape, trainable=False, initializer=tf.constant_initializer(b))
        if filter_group:
            x0, x1 = tf.split(tensor, 2, 3)
            w0, w1 = tf.split(window, 2, 3)
            h0 = tf.nn.conv2d(x0, w0, strides=[1, stride, stride, 1], padding='VALID')
            h1 = tf.nn.conv2d(x1, w1, strides=[1, stride, stride, 1], padding='VALID')
            h = tf.concat([h0, h1], 3) + b
        else:
            h = tf.nn.conv2d(tensor, window, strides=[1, stride, stride, 1], padding='VALID') + b
        if batch_norm:
            h = tf.layers.batch_normalization(h, beta_initializer=tf.constant_initializer(batch_norm_beta),
                                              gamma_initializer=tf.constant_initializer(batch_norm_gamma),
                                              moving_mean_initializer=tf.constant_initializer(batch_norm_mm),
                                              moving_variance_initializer=tf.constant_initializer(batch_norm_mv),
                                              training=False, trainable=False)

        if activation:
            h = tf.nn.relu(h)

        return h
