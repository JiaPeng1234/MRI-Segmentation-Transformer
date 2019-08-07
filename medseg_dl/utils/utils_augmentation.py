import tensorflow as tf
import math


def augment_image(images,
                  labels,
                  channels_out,
                  b_mirror=False,
                  b_rotate=False,
                  b_scale=False,
                  b_warp=False,
                  b_permute_labels=False,
                  angle_max=5,
                  scale_factor=0.05,
                  delta_max=5):

    selector = tf.random_uniform([1], minval=0, maxval=5, dtype=tf.int32)
    # selector == 0: do nothing (original image)

    images[0] = tf.Print(images[0], [tf.shape(images[0]), tf.shape(labels[0])], 'augmenting images (and labels): ')

    images = tf.stack(images, axis=0)
    labels = tf.stack(labels, axis=0)

    # mirror
    if b_mirror:
        # selector == 1
        # cond was: tf.squeeze(tf.cast(tf.random_uniform([1], minval=0, maxval=2, dtype=tf.int32), dtype=tf.bool))
        images, labels = tf.cond(tf.squeeze(tf.equal(selector, [1])),
                                 true_fn=lambda: cond_mirror(images, labels, channels_out, b_permute_labels=b_permute_labels),
                                 false_fn=lambda: (images, labels))

    # rotate
    if b_rotate:
        # selector == 2
        images, labels = tf.cond(tf.squeeze(tf.equal(selector, [2])),
                                 true_fn=lambda: cond_rotate(images, labels, angle_max),
                                 false_fn=lambda: (images, labels))

    # scale
    if b_scale:
        # selector == 3
        images, labels = tf.cond(tf.squeeze(tf.equal(selector, [3])),
                                 true_fn=lambda: cond_scale(images, labels, scale_factor),
                                 false_fn=lambda: (images, labels))

    # warp
    if b_warp:
        # selector == 4
        images, labels = tf.cond(tf.squeeze(tf.equal(selector, [4])),
                                 true_fn=lambda: cond_warp(images, labels, delta_max),
                                 false_fn=lambda: (images, labels))

    images = tf.unstack(images, axis=0)
    labels = tf.unstack(labels, axis=0)

    return images, labels


def cond_mirror(images, labels, channels_out, b_permute_labels=False):

    images = tf.unstack(images, axis=0)
    labels = tf.unstack(labels, axis=0)

    images[0] = tf.Print(images[0], [], 'selected augmentation: mirror')

    for idx in range(len(images)):
        images[idx] = tf.reverse(images[idx], axis=[0])  # atm hardcoded
    for idx in range(len(labels)):
        labels[idx] = tf.reverse(labels[idx], axis=[0])
        if b_permute_labels:
            label_hot = tf.one_hot(labels[idx], channels_out)
            # atm hardcoded label change (e.g. in symmetric case)
            label_hot = tf.concat([label_hot[..., 0:1], label_hot[..., 2:3], label_hot[..., 1:2], label_hot[..., 4:5], label_hot[..., 3:4]], axis=-1)  # permute channels
            labels[idx] = tf.argmax(label_hot, axis=-1, output_type=tf.int32)

    images = tf.stack(images, axis=0)
    labels = tf.stack(labels, axis=0)

    return images, labels


def cond_rotate(images, labels, angle_max):

    images = tf.unstack(images, axis=0)
    labels = tf.unstack(labels, axis=0)

    images[0] = tf.Print(images[0], [], 'selected augmentation: rotate')

    angle_rad_max = angle_max * math.pi / 180
    angle_rad_xy = tf.random_uniform([1], minval=-angle_rad_max, maxval=angle_rad_max, dtype=tf.float32)
    angle_rad_yz = tf.random_uniform([1], minval=-angle_rad_max, maxval=angle_rad_max, dtype=tf.float32)
    angle_rad_xz = tf.random_uniform([1], minval=-angle_rad_max, maxval=angle_rad_max, dtype=tf.float32)

    for idx in range(len(images)):
        # rotate x,y
        images[idx] = tf.contrib.image.rotate(images[idx], angle_rad_xy, interpolation='NEAREST')

        # rotate y,z
        tf.transpose(images[idx], perm=[1, 2, 0])
        images[idx] = tf.contrib.image.rotate(images[idx], angle_rad_yz, interpolation='NEAREST')
        tf.transpose(images[idx], perm=[2, 0, 1])

        # rotate x,z
        tf.transpose(images[idx], perm=[0, 2, 1])
        images[idx] = tf.contrib.image.rotate(images[idx], angle_rad_xz, interpolation='NEAREST')
        tf.transpose(images[idx], perm=[0, 2, 1])

    for idx in range(len(labels)):
        labels[idx] = tf.contrib.image.rotate(labels[idx], angle_rad_xy, interpolation='NEAREST')

        tf.transpose(labels[idx], perm=[1, 2, 0])
        labels[idx] = tf.contrib.image.rotate(labels[idx], angle_rad_yz, interpolation='NEAREST')
        tf.transpose(labels[idx], perm=[2, 0, 1])

        tf.transpose(labels[idx], perm=[0, 2, 1])
        labels[idx] = tf.contrib.image.rotate(labels[idx], angle_rad_xz, interpolation='NEAREST')
        tf.transpose(labels[idx], perm=[0, 2, 1])

    images = tf.stack(images, axis=0)
    labels = tf.stack(labels, axis=0)

    return images, labels


def cond_scale(images, labels, scale_factor):

    images = tf.unstack(images, axis=0)
    labels = tf.unstack(labels, axis=0)

    images[0] = tf.Print(images[0], [], 'selected augmentation: scale')

    # e.g with tf.contrib.image.transform
    scale = tf.squeeze(tf.random_uniform([1], minval=1 - scale_factor, maxval=1 + scale_factor, dtype=tf.float32))
    shift = tf.multiply(tf.cast(tf.shape(images[0]), dtype=tf.float32), (1-scale)/2)  # translation to keep it centered

    for idx in range(len(images)):
        # scale x,y
        images[idx] = tf.contrib.image.transform(images[idx], [scale, 0, shift[0], 0, scale, shift[1], 0, 0], interpolation='NEAREST')

        # scale z
        tf.transpose(images[idx], perm=[2, 0, 1])
        images[idx] = tf.contrib.image.transform(images[idx], [scale, 0, shift[2], 0, 1, 0, 0, 0], interpolation='NEAREST')
        tf.transpose(images[idx], perm=[1, 2, 0])

    for idx in range(len(labels)):
        labels[idx] = tf.contrib.image.transform(labels[idx], [scale, 0, shift[0], 0, scale, shift[1], 0, 0], interpolation='NEAREST')

        tf.transpose(labels[idx], perm=[2, 0, 1])
        labels[idx] = tf.contrib.image.transform(labels[idx], [scale, 0, shift[2], 0, 1, 0, 0, 0], interpolation='NEAREST')
        tf.transpose(labels[idx], perm=[1, 2, 0])

    images = tf.stack(images, axis=0)
    labels = tf.stack(labels, axis=0)

    return images, labels


def cond_warp(images, labels, delta_max):

    raise NotImplementedError('This function hasn\'t been fully implemented yet')

    images[0] = tf.Print(images[0], [], 'selected augmentation: warp')
    # e.g. with sparse_image_warp, dense_image_warp
    # random array deltas with something like random state rand?
    # TODO: introduce gaussian filter to regularize shift,
    # TODO: may require tf.nn.conv3d
    '''    d = tf.distributions.Normal(mean, std)

    vals = d.prob(tf.range(start = -size, limit = size + 1, dtype = tf.float32))

    gauss_kernel = tf.einsum('i,j->ij',
                                  vals,
                                  vals)'''

    deltas = tf.random_uniform(tf.shape(image) + [3], minval=-delta_max, maxval=delta_max, dtype=tf.float32)
    # warp x,y
    image = tf.expand_dims(image, axis=3)
    image = tf.transpose(image, perm=[2, 0, 1, 3])  # depth is becoming a "batch" so transform can be applied
    image = tf.contrib.image.dense_image_warp(image, tf.transpose(deltas[..., :3], perm=[2, 0, 1, 3]))

    # warp z
    image = tf.transpose(image, perm=[1, 0, 2, 3])  # return x, z, y, channel
    image = tf.contrib.image.dense_image_warp(image, tf.transpose(tf.concat([deltas[..., 3:4], tf.zeros_like(deltas[..., 3:4])], axis=-1),
                                                                  perm=[0, 2, 1, 3]))
    image = tf.squeeze(tf.transpose(image, perm=[0, 2, 1, 3]))

    return image, label
