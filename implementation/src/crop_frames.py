import tensorflow as tf

from src.parse_arguments import parameters


# Pad image (with black or image mean) with a variable amount of pixels
# to assure a window placed in bounding box center is always included in the padded image
def frame_padding(image, bbox_x, bbox_y, window_size):
    image_size = tf.shape(image)

    window_center = window_size / 2

    left_padding = tf.maximum(0, -tf.cast(tf.round(bbox_x - window_center), tf.int32))
    top_padding = tf.maximum(0, -tf.cast(tf.round(bbox_y - window_center), tf.int32))
    right_padding = tf.maximum(0, tf.cast(tf.round(bbox_x + window_center), tf.int32) - image_size[1])
    bottom_padding = tf.maximum(0, tf.cast(tf.round(bbox_y + window_center), tf.int32) - image_size[0])

    max_padding = tf.reduce_max([left_padding, top_padding, right_padding, bottom_padding])
    padding = [[max_padding, max_padding], [max_padding, max_padding], [0, 0]]

    image_mean = tf.reduce_mean(image, axis=(0, 1), name='image_mean') if parameters.design.pad_with_image_mean else 0

    image = image - image_mean
    image = tf.pad(image, padding, mode='CONSTANT')
    image = image + image_mean

    return image, max_padding


# For every window size in window sizes
# Crop an image of window size centering it in bounding box coordinates, then resizing it to network size
# Images are finally stacked together ( one image has shape (1,w,h,c) )
def crop_resize(image, padding, bbox_x, bbox_y, window_sizes, network_size):

    images = []

    for window_size in window_sizes:

        window_center = window_size / 2

        # Top-left corner of bounding box
        bbox_top_left_x = padding + tf.cast(tf.round(bbox_x - window_center), tf.int32)
        bbox_top_left_y = padding + tf.cast(tf.round(bbox_y - window_center), tf.int32)

        cropped_image = tf.image.crop_to_bounding_box(image,
                                             tf.cast(bbox_top_left_y, tf.int32),
                                             tf.cast(bbox_top_left_x, tf.int32),
                                             tf.cast(window_size, tf.int32),
                                             tf.cast(window_size, tf.int32))

        resized_image = tf.image.resize_images(cropped_image, [network_size, network_size], method=tf.image.ResizeMethod.BILINEAR)

        images.append(resized_image)

    images = tf.stack(images)

    return images


'''
Test

tf.enable_eager_execution()

image = tf.read_file('prova.jpg')
image = tf.image.decode_image(image)
image, padding = frame_padding(image, 112, 112, 100)
images = crop_resize(image, padding, 112, 112, [25, 50, 100], 200)

i = 0
for image in images:
    image = tf.cast(image, tf.uint8)
    image = tf.image.encode_jpeg(image, quality=100)
    tf.write_file('result'+str(i)+'.jpg', image)
    i += 1

'''



