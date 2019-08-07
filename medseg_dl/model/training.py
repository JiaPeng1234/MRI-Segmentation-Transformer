import tensorflow as tf
import os
import logging
from medseg_dl.utils import utils_misc


# from tqdm import tqdm


def sess_train(spec_pipeline, spec_model, params):

    # Add an op to initialize the variables
    init_op_vars = tf.global_variables_initializer()

    # Fetch global step of default graph
    global_step = tf.train.get_global_step()

    # Add ops to save and restore all variables
    saver_recent = tf.train.Saver(max_to_keep=10)  # keeps the last 10 ckpts

    # generate summary writer
    writer = tf.summary.FileWriter(params.dict['dir_logs_train'])
    logging.info(f'saving log to {params.dict["dir_logs_train"]}')

    # Define fetched variables
    fetched_train = {'loss_value': spec_model['loss'],
                     'train_op': spec_model['train_op'],
                     'update_metrics_op_train': spec_model['update_op_metrics'],
                     'summary_train': spec_model['summary_op_train'],
                     'gstep': global_step}

    fetched_metrics_train = {'metrics': spec_model['metrics_values'],
                             'summary_metrics': spec_model['summary_op_metrics']}

    if params.dict['b_viewer_train']:
        fetched_train.update({'images': spec_pipeline['images'],
                              'labels': spec_pipeline['labels'],
                              'positions': spec_pipeline['positions'],
                              'probs': spec_model['probs']})

    # set growth option
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        _ = tf.summary.FileWriter(params.dict['dir_graphs_train'], sess.graph)
        logging.info(f'Graph saved in {params.dict["dir_graphs_train"]}')

        sess.run(init_op_vars)  # init global variables
        first_epoch = 0
        total_steps = 0

        # Reload weights from directory if previous training is restored
        if params.dict['b_restore']:
            if os.path.isdir(params.dict['dir_ckpts']):
                file_ckpt = tf.train.latest_checkpoint(params.dict['dir_ckpts'])
                first_epoch = int(os.path.basename(file_ckpt).split('-')[1]) + 1
                logging.info(f'Restoring parameters from {file_ckpt}')
                saver_recent.restore(sess, file_ckpt)

        # Epochs
        for epoch in range(first_epoch, params.dict['num_epochs']):

            # training
            logging.info(f'Epoch {epoch + 1}/{params.dict["num_epochs"]}: training')
            sess.run(spec_pipeline['init_op_iter'])  # initialize dataset
            sess.run(spec_model['init_op_metrics'])  # reset metrics

            # training step
            #pbar = tqdm(total=total_steps)
            while True:
                try:
                    results = sess.run(fetched_train)  # perform mini-batch update
                    gstep = results['gstep']
                    loss_value = results['loss_value']
                    logging.info(f'Step {gstep}, loss: {loss_value}')
                    writer.add_summary(results['summary_train'], global_step=gstep)  # write loss per batch

                    # allow viewing of data
                    if params.dict['b_viewer_train']:
                        shown_index = 2
                        print(f'Visualized patch has a position of {results["positions"][shown_index,...]}')
                        utils_misc.show_results(results['images'][shown_index, ...], results['labels'][shown_index, ...], results['probs'][shown_index, ...])
                    #pbar.update(1)
                except tf.errors.OutOfRangeError:
                    #pbar.close()
                    break

            # fetch aggregated metrics values
            results = sess.run(fetched_metrics_train)
            writer.add_summary(results['summary_metrics'], global_step=epoch)  # write metrics per epoch

            # save model every epoch:
            # if epoch % 5 == 0:
            save_path = saver_recent.save(sess,
                                          os.path.join(params.dict['dir_ckpts'], 'model.ckpt'),
                                          global_step=epoch)
            logging.info(f'Model saved in {save_path}')
