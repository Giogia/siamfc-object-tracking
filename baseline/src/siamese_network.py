import os.path

import numpy as np

from src.crop_frames import *

MODEL_PATH = os.path.join(parameters.environment.model_folder, parameters.environment.model_name)

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
frame = tf.placeholder(tf.uint8, name='frame')

window_size_z_ph = tf.placeholder(tf.float64, name="window_size")
window_size_x_0_ph = tf.placeholder(tf.float64)
window_size_x_1_ph = tf.placeholder(tf.float64)
window_size_x_2_ph = tf.placeholder(tf.float64)


def build_tracking_graph():

    # Turn image into a Tensor
    image = 255.0 * tf.image.convert_image_dtype(frame, tf.float64)

    # Pad frames if necessary
    padded_frame_z, padding_z = frame_padding(image, bbox_x_ph, bbox_y_ph, window_size_z_ph)
    padded_frame_x, padding_x = frame_padding(image, bbox_x_ph, bbox_y_ph, window_size_x_2_ph)

    # Extract tensor Z
    tensor_z = crop_resize(padded_frame_z, padding_z, bbox_x_ph, bbox_y_ph, [window_size_z_ph],
                           parameters.design.exemplar_sz)

    # Extract tensor X (3 scales)
    tensor_x = crop_resize(padded_frame_x, padding_x, bbox_x_ph, bbox_y_ph,
                           [window_size_x_0_ph, window_size_x_1_ph, window_size_x_2_ph],
                           parameters.design.search_sz)

    network_z, network_x = create_network(tensor_z, tensor_x)

    network_z = tf.squeeze(network_z)
    network_z = tf.stack([network_z, network_z, network_z])

    # Compare templates via cross-correlation
    scores = cross_correlation(network_z, network_x)

    final_score_size = parameters.hyperparameters.response_up * (parameters.design.score_sz - 1) + 1

    upsampled_scores = tf.image.resize_images(scores, [final_score_size, final_score_size],
                                              method=tf.image.ResizeMethod.BICUBIC, align_corners=True)

    return image, network_z, upsampled_scores


def create_network(network_z, network_x):
    for i in range(NUM_LAYERS):

        print('> Layer ' + str(i + 1))

        conv_w_name = 'conv' + str(i + 1) + '/W'
        conv_b_name = 'conv' + str(i + 1) + '/b'

        print('\t\tCONV: setting ' + conv_w_name + ' ' + conv_b_name)
        print('\t\tCONV: stride ' + str(CONV_STRIDE[i]) + ', filter-group ' + str(FILTERGROUP[i]))

        conv_w = tf.train.load_variable(ckpt_dir_or_file=MODEL_PATH, name=conv_w_name)
        conv_b = tf.train.load_variable(ckpt_dir_or_file=MODEL_PATH, name=conv_b_name)

        if BNORM[i]:
            bn_beta_name = 'conv' + str(i + 1) + '/batch_normalization/beta'
            bn_gamma_name = 'conv' + str(i + 1) + '/batch_normalization/gamma'
            bn_moving_mean_name = 'conv' + str(i + 1) + '/batch_normalization/moving_mean'
            bn_moving_variance_name = 'conv' + str(i + 1) + '/batch_normalization/moving_variance'

            print(
                '\t\tBNORM: setting ' + bn_beta_name + ' ' + bn_gamma_name + ' ' + bn_moving_mean_name + ' ' + bn_moving_variance_name)

            bn_beta = tf.train.load_variable(ckpt_dir_or_file=MODEL_PATH, name=bn_beta_name)
            bn_gamma = tf.train.load_variable(ckpt_dir_or_file=MODEL_PATH, name=bn_gamma_name)
            bn_moving_mean = tf.train.load_variable(ckpt_dir_or_file=MODEL_PATH, name=bn_moving_mean_name)
            bn_moving_variance = tf.train.load_variable(ckpt_dir_or_file=MODEL_PATH, name=bn_moving_variance_name)

        else:
            bn_beta = bn_gamma = bn_moving_mean = bn_moving_variance = []

        # Reuse=True for Siamese parameters sharing
        network_z = create_convolutional(network_z, conv_w, conv_b, CONV_STRIDE[i],
                                         bn_beta, bn_gamma, bn_moving_mean, bn_moving_variance,
                                         filter_group=FILTERGROUP[i], batch_norm=BNORM[i], activation=RELU[i],
                                         scope='conv' + str(i + 1), reuse=True)

        # Set up convolutional block with batch normalization and activation
        network_x = create_convolutional(network_x, conv_w, conv_b, CONV_STRIDE[i],
                                         bn_beta, bn_gamma, bn_moving_mean, bn_moving_variance,
                                         filter_group=FILTERGROUP[i], batch_norm=BNORM[i], activation=RELU[i],
                                         scope='conv' + str(i + 1), reuse=False)

        # Add max pool if required
        if POOL_STRIDE[i] > 0:
            print('\t\tMAX-POOL: size {} and stride {}'.format(POOL_SIZE, POOL_STRIDE[i]))
            network_x = tf.nn.max_pool(network_x, [1, POOL_SIZE, POOL_SIZE, 1], strides=[1, POOL_STRIDE[i], POOL_STRIDE[i], 1],
                                   padding='VALID', name='pool' + str(i + 1))
            network_z = tf.nn.max_pool(network_z, [1, POOL_SIZE, POOL_SIZE, 1], strides=[1, POOL_STRIDE[i], POOL_STRIDE[i], 1],
                                   padding='VALID', name='pool' + str(i + 1))

    return network_z, network_x


def create_convolutional(tensor, window, bias, stride, batch_norm_beta, batch_norm_gamma, batch_norm_mm, batch_norm_mv,
                         filter_group=False, batch_norm=True, activation=True, scope="conv", reuse=False):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):

        # sanity check
        # window = tf.get_variable("W", window.shape, trainable=False, initializer=tf.constant_initializer(window))
        # bias = tf.get_variable("b", bias.shape, trainable=False, initializer=tf.constant_initializer(bias))

        if filter_group:
            x0, x1 = tf.split(tensor, 2, 3)
            w0, w1 = tf.split(window, 2, 3)
            h0 = tf.nn.conv2d(x0, w0, strides=[1, stride, stride, 1], padding='VALID')
            h1 = tf.nn.conv2d(x1, w1, strides=[1, stride, stride, 1], padding='VALID')
            layer = tf.concat([h0, h1], 3) + bias

        else:
            layer = tf.nn.conv2d(tensor, window, strides=[1, stride, stride, 1], padding='VALID') + bias

        if batch_norm:
            layer = tf.layers.batch_normalization(layer, beta_initializer=tf.constant_initializer(batch_norm_beta),
                                                  gamma_initializer=tf.constant_initializer(batch_norm_gamma),
                                                  moving_mean_initializer=tf.constant_initializer(batch_norm_mm),
                                                  moving_variance_initializer=tf.constant_initializer(batch_norm_mv),
                                                  training=False, trainable=False)

        if activation:
            layer = tf.nn.relu(layer)

        return layer


def cross_correlation(network_z, network_x):
    # z, x are [batch, height, width, channels]
    network_z = tf.transpose(network_z, perm=[1, 2, 0, 3])
    network_x = tf.transpose(network_x, perm=[1, 2, 0, 3])
    # z, x are [height, width, batch, channels]
    hz, wz, bz, cz = tf.unstack(tf.shape(network_z))
    hx, wx, bx, cx = tf.unstack(tf.shape(network_x))

    # assert b==bx, ('Z and X should have same Batch size')
    # assert c==cx, ('Z and X should have same Channels number')
    network_z = tf.reshape(network_z, (hz, wz, bz * cz, 1))
    network_x = tf.reshape(network_x, (1, hx, wx, bz * cz))

    final_network = tf.nn.depthwise_conv2d(network_x, network_z, strides=[1, 1, 1, 1], padding='VALID')
    # final is [1, hf, wf, bc]
    final_network = tf.concat(tf.split(final_network, 3, axis=3), axis=0)
    # final is [b, hf, wf, c]
    final_network = tf.expand_dims(tf.reduce_sum(final_network, axis=3), axis=3)
    # final is [b, hf, wf, 1]

    if BNORM_ADJUST:
        bn_beta = tf.train.load_variable(ckpt_dir_or_file=MODEL_PATH, name='batch_normalization/beta')
        bn_gamma = tf.train.load_variable(ckpt_dir_or_file=MODEL_PATH, name='batch_normalization/gamma')
        bn_moving_mean = tf.train.load_variable(ckpt_dir_or_file=MODEL_PATH, name='batch_normalization/moving_mean')
        bn_moving_variance = tf.train.load_variable(ckpt_dir_or_file=MODEL_PATH, name='batch_normalization/moving_variance')

        final_network = tf.layers.batch_normalization(final_network, beta_initializer=tf.constant_initializer(bn_beta),
                                                      gamma_initializer=tf.constant_initializer(bn_gamma),
                                                      moving_mean_initializer=tf.constant_initializer(bn_moving_mean),
                                                      moving_variance_initializer=tf.constant_initializer(bn_moving_variance),
                                                      training=False, trainable=False)

    return final_network
