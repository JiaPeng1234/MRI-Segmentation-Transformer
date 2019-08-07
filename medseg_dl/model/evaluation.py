import tensorflow as tf
import os
import logging
from medseg_dl.utils import utils_misc
import numpy as np
import datetime


def sess_eval(spec_pipeline, spec_pipeline_metrics, spec_model, params, filenames_eval=''):

    # Add an op to initialize the variables
    init_op_vars = tf.global_variables_initializer()

    # Add ops to save and restore all variables
    saver_best = tf.train.Saver(max_to_keep=1)  # only keep best checkpoint

    # generate summary writer
    writer = tf.summary.FileWriter(params.dict['dir_logs_eval'])
    logging.info(f'saving log to {params.dict["dir_logs_eval"]}')

    # Define fetched variables
    fetched_eval = {'agg_probs': spec_model['agg_probs'],
                    'recombined_probs_op': spec_model['recombined_probs_op']}

    fetched_metrics_op = {'recombined_probs': spec_model['recombined_probs_value'],
                          'update_metrics_op_eval': spec_model['update_op_metrics']}

    fetched_metrics_eval = {'metrics': spec_model['metrics_values'],
                            'summary_metrics': spec_model['summary_op_metrics']}

    if params.dict['b_viewer_eval']:
        fetched_metrics_op.update({'images': spec_pipeline_metrics['images'],
                                   'labels': spec_pipeline_metrics['labels']})  # 'probs': spec_model['probs']})

    # set growth option
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        _ = tf.summary.FileWriter(params.dict['dir_graphs_eval'], sess.graph)
        logging.info(f'Graph saved in {params.dict["dir_graphs_eval"]}')

        sess.run(init_op_vars)  # init global variables
        best_eval_acc = 0

        if params.dict['b_continuous_eval']:
            # Run evaluation when there"s a new checkpoint
            logging.info(f'Continuous evaluation of {params.dict["dir_ckpts"]}')
            for ckpt in tf.contrib.training.checkpoints_iterator(params.dict['dir_ckpts'],
                                                                 min_interval_secs=30,
                                                                 timeout=3600,
                                                                 timeout_fn=timeout_fn):
                logging.info('Processing new checkpoint')
                try:
                    results, epoch = eval_epoch(sess=sess,
                                                ckpt=ckpt,
                                                saver=saver_best,
                                                spec_pipeline=spec_pipeline,
                                                spec_pipeline_metrics=spec_pipeline_metrics,
                                                spec_model=spec_model,
                                                fetched_eval=fetched_eval,
                                                fetched_metrics_op=fetched_metrics_op,
                                                fetched_metrics_eval=fetched_metrics_eval,
                                                writer=writer,
                                                params=params,
                                                filenames=filenames_eval)

                    # If best_eval, best_save_path
                    eval_acc = results['metrics']['mean_iou']
                    if eval_acc >= best_eval_acc:
                        # Store new best accuracy
                        logging.info(f'Found new best metric, new: {eval_acc}, old: {best_eval_acc}')
                        best_eval_acc = eval_acc

                        # Save weights
                        save_path = saver_best.save(sess,
                                                    os.path.join(params.dict['dir_ckpts_best'], 'model.ckpt'),
                                                    global_step=epoch)
                        logging.info(f'Best model saved in {save_path}')

                        # Save best eval metrics in a json file in the model directory
                        metrics_path_best = os.path.join(params.dict['dir_model'], "metrics_eval_best.yaml")
                        utils_misc.save_dict_to_yaml(results['metrics'], metrics_path_best)

                    # check if max amount of checkpoints is reached
                    if epoch >= params.dict['num_epochs']:
                        tf.logging.info(f'Evaluation finished after epoch {epoch}')
                        break

                except tf.errors.NotFoundError:  # Note: this is sometimes reached if training has already finished
                    logging.info(f'Checkpoint {ckpt} no longer exists, skipping checkpoint')

        else:
            # Run evaluation on most recent checkpoint
            logging.info(f'Single evaluation of {params.dict["dir_ckpts"]}')
            ckpt = tf.train.latest_checkpoint(params.dict['dir_ckpts'])

            _, _ = eval_epoch(sess=sess,
                              ckpt=ckpt,
                              saver=saver_best,
                              spec_pipeline=spec_pipeline,
                              spec_pipeline_metrics=spec_pipeline_metrics,
                              spec_model=spec_model,
                              fetched_eval=fetched_eval,
                              fetched_metrics_op=fetched_metrics_op,
                              fetched_metrics_eval=fetched_metrics_eval,
                              writer=writer,
                              params=params,
                              filenames=filenames_eval)


def eval_epoch(sess, ckpt, saver, spec_pipeline, spec_pipeline_metrics, spec_model, fetched_eval, fetched_metrics_op, fetched_metrics_eval, writer, params, filenames=''):

    epoch = int(os.path.basename(ckpt).split('-')[1])
    logging.info(f'Epoch {epoch}: evaluation')
    saver.restore(sess, ckpt)
    logging.info(f'Epoch {epoch}: restored checkpoint')

    sess.run(spec_model['init_op_metrics'])  # reset metrics

    # process all eval batches per evaluation subject
    for idx_subject in range(len(filenames[0][0])):
        logging.info(f'Processing subject {idx_subject}/{len(filenames[0][0])}')

        # initialize dataset for patches
        sess.run(spec_pipeline['init_op_iter'], feed_dict={spec_pipeline['idx_selection']: idx_subject})
        sess.run(spec_model['agg_probs_init_op'])  # initialize aggregated probs tensor and batch count

        # aggregate patches
        results = None
        while True:
            try:
                results = sess.run(fetched_eval)
            except tf.errors.OutOfRangeError:
                break

        logging.info(f'Epoch {epoch}: fetching metrics')
        # initialize dataset for metric calculation (i.e. no patches)
        sess.run(spec_pipeline_metrics['init_op_iter'], feed_dict={spec_pipeline_metrics['idx_selection']: idx_subject})
        results_op = sess.run(fetched_metrics_op)

        # save prediction
        if params.dict['b_save_pred']:
            now = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
            path_save = '/home/d1280/no_backup/d1280/results'
            subject_name = os.path.basename(os.path.dirname(os.path.normpath(filenames[0][0][idx_subject])))
            np.save(os.path.join(path_save, str(params.dict['idx_dataset']), subject_name + '_' + now + '_images'), results_op['images'])
            np.save(os.path.join(path_save, str(params.dict['idx_dataset']), subject_name + '_' + now + '_labels'), results_op['labels'])
            np.save(os.path.join(path_save, str(params.dict['idx_dataset']), subject_name + '_' + now + '_preds'), results_op['recombined_probs'])

        # allow viewing of data
        if params.dict['b_viewer_eval']:
            utils_misc.show_results(results_op['images'][0, ...], results_op['labels'][0, ...], results_op['recombined_probs'][0, ...])

    results_metrics = sess.run(fetched_metrics_eval)
    logging.info(f'Epoch {epoch}: fetched metrics: {results_metrics["metrics"]}')
    writer.add_summary(results_metrics['summary_metrics'], global_step=epoch)

    # Save latest eval metrics in a json file in the model directory
    metrics_path_last = os.path.join(params.dict['dir_model'], "metrics_eval_last.yaml")
    utils_misc.save_dict_to_yaml(results_metrics['metrics'], metrics_path_last)

    return results_metrics, epoch


def timeout_fn():

    logging.info('No new checkpoint: assuming training has ended')

    return True
