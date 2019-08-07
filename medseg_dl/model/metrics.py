import tensorflow as tf
import os


def metrics_fn(labels, probs, channels_out, b_verbose=False):
    """ calc metrics and fetch ops """

    labels_var = tf.argmax(labels, axis=-1)
    labels_hot = tf.one_hot(labels_var, channels_out)
    predictions_var = tf.argmax(probs, axis=-1)
    predictions_hot = tf.one_hot(predictions_var, channels_out)
    metrics = fetch_metrics(labels_hot, predictions_hot, labels_var, predictions_var, channels_out)

    if b_verbose:
        metrics = tf.Print(metrics, [tf.shape(labels_hot), tf.shape(predictions_hot)], 'fetched metrics with labels/preds: ', summarize=20)

    # Get the values of the metrics (used for update later)
    metrics_values = {k: v[0] for k, v in metrics.items()}

    # Group the update ops for the tf.metrics, so that we can run only one op to update them all
    update_metrics_op = tf.group(*[op for _, op in metrics.values()])

    # Get the op to reset the local variables used in tf.metrics, for when we restart an epoch
    scope_full = os.path.join(tf.get_default_graph().get_name_scope())
    metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope=scope_full)
    metrics_init_op = tf.variables_initializer(metric_variables)

    return metrics_init_op, update_metrics_op, metrics_values


def fetch_metrics(labels_hot, predictions_hot, labels_var, predictions_var, channels_out):
    """ metrics used for calculation """

    dice_scores = calc_dice_scores(labels_hot, predictions_hot)
    single_dice_scores = tf.split(dice_scores, channels_out, axis=-1)

    # generate masks for proper mean scores
    mean_masks = tf.clip_by_value(tf.count_nonzero(labels_hot, axis=[1, 2, 3]), 0, 1)
    masks = tf.split(mean_masks, channels_out, axis=-1)
    mean_dice_scores = tf.reduce_mean(tf.boolean_mask(dice_scores, mean_masks), axis=-1)

    # average metrics
    metrics = {
            'accuracy': tf.metrics.accuracy(labels_hot, predictions_hot),
            'mean_pc_acc': tf.metrics.mean_per_class_accuracy(labels_var, predictions_var, 3),
            'mean_iou': tf.metrics.mean_iou(labels_hot, predictions_hot, 3),
            'mean_dice': tf.metrics.mean(mean_dice_scores)}

    # single class metrics
    for idx_ch in range(channels_out):
        metrics[f'dice_c{idx_ch}'] = tf.metrics.mean(single_dice_scores[idx_ch], weights=masks[idx_ch])

    return metrics


def calc_dice_scores(labels_hot, predictions_hot, eps=1e-12):

    nom = 2 * tf.reduce_sum(tf.cast(tf.logical_and(tf.cast(labels_hot, dtype=tf.bool), tf.cast(predictions_hot, dtype=tf.bool)), dtype=tf.float32), axis=(1, 2, 3))
    denom = tf.reduce_sum(labels_hot + predictions_hot, axis=(1, 2, 3))
    scores = tf.divide(nom, denom + eps)

    return scores
