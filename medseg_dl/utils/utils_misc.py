import logging
import yaml
import numpy as np
import tensorflow as tf


def set_logger(log_path, log_level):

    #
    logger = logging.getLogger()

    # Logging to a file
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logger.addHandler(file_handler)

    if not logger.handlers:
        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)

    logger.setLevel(log_level)


def set_tf_logger(log_level):

    tf.logging.set_verbosity(log_level)


def save_dict_to_yaml(d, path_yaml):
    with open(path_yaml, 'w') as file:
        # We need to convert the values to float for yaml (it doesn't accept np.array, np.float, )
        # TODO: check if this is the case for yaml
        d = {k: float(v) for k, v in d.items()}
        yaml.dump(d, file, indent=4)


def show_results(images, labels, probs):
    """ takes array and plots it """
    import nibabel
    shape_img = images[..., 0].shape
    shape_label = labels[..., 0].shape

    if any(np.greater(shape_img, shape_label)):
        pad_width = [int((shape_img[x] - shape_label[x]) / 2) for x in range(3)]
        labels_fit = np.pad(labels,
                            ((pad_width[0],), (pad_width[1],), (pad_width[2],), (0,)),
                            'constant')
        probs_fit = np.pad(probs,
                           ((pad_width[0],), (pad_width[1],), (pad_width[2],), (0,)),
                           'constant')
    else:
        labels_fit = labels
        probs_fit = probs

    images = np.stack((*[np.squeeze(x) for x in np.split(images, images.shape[-1], axis=-1)],
                       np.argmax(labels_fit, axis=-1),
                       np.argmax(probs_fit, axis=-1)), axis=3)

    nibabel.viewers.OrthoSlicer3D(images).show()
