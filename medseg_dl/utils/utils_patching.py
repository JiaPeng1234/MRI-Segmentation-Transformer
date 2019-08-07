import tensorflow as tf
import logging


def space_to_batch(images, labels, tiles, n_tiles, paddings_image, paddings_tiles, shape_padded_image, shape_padded_label, shape_input, shape_output, b_with_labels=False, b_verbose=False):
    """ takes stack (tensors) of images and labels and processes them into patches stacked as a batch
        atm hardcoded to shape_img, otherwise one has to use pyfunc or dynamic while loops
    """

    # map parse function to each zipped element
    print(paddings_tiles, shape_padded_label, shape_output)
    assert any([a % b <= 0 for a, b in zip(shape_padded_label, shape_output)])

    paddings_both = [a + b for a, b in zip(paddings_image, paddings_tiles)]
    shape_padded_both = [a + 2 * b for a, b in zip(shape_padded_image, paddings_tiles)]
    scale_factor = [float(a/b) for a, b in zip(shape_padded_both, shape_padded_image)]

    paddings_labels = [(x, x) for x in paddings_tiles] + [(0, 0)]
    paddings_both = [(x, x) for x in paddings_both] + [(0, 0)]

    if b_verbose:
        print('Padding/ padding_img: ', paddings_labels, paddings_both, scale_factor)
    logging.info('Using %d patches to predict a whole image', n_tiles)

    # process labels into patches
    if b_with_labels:
        # print('labels prior: ', labels)
        labels = tf.pad(labels, paddings_labels)
        labels = tf.expand_dims(labels, axis=0)
        batch_shape = tf.stack([n_tiles, *shape_output, tf.shape(labels)[-1]])
        labels = tf.reshape(labels, batch_shape)
        # print('labels post: ', labels)

    # process images into patches
    # Note: a simple reshape is not possible due to the overlapping of inputs
    #       map_fn or tf while_loops or sth similar might help
    images = tf.pad(images, paddings_both)
    if b_verbose:
        images = tf.Print(images, [tf.shape(images), tiles], 'Temporary patch shape - before: ', summarize=5)

    patches = [None for _ in range(n_tiles)]
    # patch_indices = list(range(n_tiles))
    positions = [None for _ in range(n_tiles)]
    offset_image = [int(x / 2) for x in shape_input]
    idx_tile = 0
    for idx_0 in range(tiles[0]):
        for idx_1 in range(tiles[1]):
            for idx_2 in range(tiles[2]):
                start_pos = [shape_output[0] * idx_0, shape_output[1] * idx_1, shape_output[2] * idx_2, 0]
                positions[idx_tile] = [float(a + b) for a, b in zip(start_pos[0:3], offset_image)]
                patches[idx_tile] = tf.slice(images, start_pos, shape_input + [tf.shape(images)[-1]])
                idx_tile += 1
                # images = tf.Print(images, [tf.shape(images), idx_0, idx_1, idx_2, start_pos], 'performed crop at: ')

    if b_verbose:
        patches[0] = tf.Print(patches[0], [tf.shape(patches[0])], 'Temporary patch shape - within: ', summarize=5)
    images = tf.stack(patches, axis=0)

    positions_t = tf.stack(positions, axis=0)
    positions_t = tf.cast(tf.multiply((tf.divide(positions_t, shape_padded_both) - 0.5) * 2, scale_factor), dtype=tf.float32)  # rescale it | account for larger padded size
    if b_verbose:
        images = tf.Print(images, [tf.shape(images)], 'Temporary patch shape - after: ', summarize=5)

    return images, labels, positions_t


def batch_to_space(stacked_patches, tiles, shape_padded_label, shape_image, channels, b_verbose=False):
    """ takes list of predictions and processes them into a full image """

    shape_stacked = tf.unstack(tf.shape(stacked_patches))  # something like [x, img_1, img_2, img_3, channel]
    stacked_patches = tf.reshape(stacked_patches, [tiles[0], tiles[1], tiles[2], *shape_stacked[1:]])  # split stacks into tiles
    stacked_patches = tf.transpose(stacked_patches, perm=[0, 3, 1, 4, 2, 5, 6])  # interleave tiles and img dims
    if b_verbose:
        stacked_patches = tf.Print(stacked_patches, [tf.shape(stacked_patches)], 'stacked_patches:', summarize=10)
    image = tf.reshape(stacked_patches, [-1, *shape_padded_label, channels])  # reshape into proper image
    if b_verbose:
        image = tf.Print(image, [tf.shape(image)], 'new_image:', summarize=10)

    # crop image to final size
    pos_begin = [int((a - b) / 2) for a, b in zip(shape_padded_label, shape_image)]
    image = tf.Print(image, [tf.shape(image), pos_begin, shape_image, channels], 'shapes before slicing', summarize=5)
    image = tf.slice(image,
                     [0, *pos_begin, 0],
                     [1, *shape_image, channels])

    return image
