import tensorflow as tf
import numpy as np


def set_convolutional(x, w, b, stride, bn_beta, bn_gamma, bn_mm, bn_mv, filter_group=False, batch_norm=True,
                      activation=True, scope=None, reuse=False):
    # use the input scope or default to "conv"
    with tf.variable_scope(scope or 'conv', reuse=reuse):
        # sanity check    
        # w = tf.get_variable("W", w.shape, trainable=False, initializer=tf.constant_initializer(w))
        # b = tf.get_variable("b", b.shape, trainable=False, initializer=tf.constant_initializer(b))
        if filter_group:
            x0, x1 = tf.split(x, 2, 3)
            w0, w1 = tf.split(w, 2, 3)
            h0 = tf.nn.conv2d(x0, w0, strides=[1, stride, stride, 1], padding='VALID')
            h1 = tf.nn.conv2d(x1, w1, strides=[1, stride, stride, 1], padding='VALID')
            h = tf.concat([h0, h1], 3) + b
        else:
            h = tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding='VALID') + b
        if batch_norm:
            bn_beta = np.squeeze(bn_beta)
            bn_gamma = np.squeeze(bn_gamma)
            #print(bn_beta.shape, bn_gamma.shape, bn_mm.shape, bn_mv.shape)
            h = tf.layers.batch_normalization(h, beta_initializer=tf.constant_initializer(bn_beta),
                                              gamma_initializer=tf.constant_initializer(bn_gamma),
                                              moving_mean_initializer=tf.constant_initializer(bn_mm),
                                              moving_variance_initializer=tf.constant_initializer(bn_mv),
                                              training=False, trainable=False)

        if activation:
            h = tf.nn.relu(h)

        return h
