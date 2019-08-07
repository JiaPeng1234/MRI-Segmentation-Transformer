import tensorflow as tf


def soft_dice(labels, probs, eps=1e-12):

    float_labels = tf.cast(labels, tf.float32)
    nom = 2 * tf.reduce_sum(tf.multiply(float_labels, probs), axis=(1, 2, 3))
    denom = tf.reduce_sum(tf.add(float_labels, tf.square(probs)), axis=(1, 2, 3))
    loss = 1 - tf.reduce_mean(tf.divide(nom, tf.add(denom, eps)))

    return loss


def soft_jaccard(labels, probs, smooth=1e-6):

    float_labels = tf.cast(labels, tf.float32)
    intersec = tf.reduce_sum(tf.multiply(float_labels, probs), axis=(1, 2, 3))
    sum_ = tf.reduce_sum(float_labels + probs, axis=(1, 2, 3))
    jaccard = (intersec + smooth) / (sum_ - intersec + smooth)
    loss = 1 - tf.reduce_mean(jaccard)
    # loss = 1 - tf.divide(tf.reduce_sum(jaccard), tf.cast(tf.size(jaccard), dtype=tf.float32))

    return loss
