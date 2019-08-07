import os
import logging
import tensorflow as tf
from medseg_dl import parameters
from medseg_dl.utils import utils_data, utils_misc
from medseg_dl.model import input_fn
from medseg_dl.model import model_fn
from medseg_dl.model import training
import sys


def main(dir_model, non_local='disable', attgate='disable', device=None, idx_dataset=0):
    # Since this is the training script all variables should be trainable
    b_training = True  # hardcoded since scripts are not identical

    # Load / generate parameters from model file, if available
    file_params = os.path.join(dir_model, 'params.yaml')
    if os.path.isfile(file_params):
        params = parameters.Params(path_yaml=file_params)
    else:
        params = parameters.Params(model_dir=dir_model, idx_dataset=idx_dataset)

    # Set logger
    utils_misc.set_logger(os.path.join(params.dict['dir_model'], 'train.log'), params.dict['log_level'])

    # Set random seed
    if params.dict['b_use_seed']:
        tf.set_random_seed(params.dict['random_seed'])

    # Set device for graph calc
    if device:
        os.environ['CUDA_VISIBLE_DEVICES'] = device
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = params.dict['device']

    """ Fetch data, generate pipeline and model """
    tf.reset_default_graph()

    # Fetch datasets, atm. saved as json
    logging.info('Fetching the datasets...')
    filenames_train, filenames_eval = utils_data.load_sets(params.dict['dir_data'],
                                                           params.dict['dir_model'],
                                                           path_parser_cfg=params.dict['path_parser_cfg'],
                                                           set_split=params.dict['set_split'],
                                                           b_recreate=True)

    # Create a tf.data pipeline
    logging.info('Creating the pipeline...')
    spec_pipeline = input_fn.gen_pipeline_train(filenames=filenames_train,
                                                shape_image=params.dict['shape_image'],
                                                shape_input=params.dict['shape_input'],
                                                shape_output=params.dict['shape_output'],
                                                channels_out=params.dict['channels_out'],
                                                size_batch=params.dict['size_batch'],
                                                size_buffer=params.dict['size_buffer'],
                                                num_parallel_calls=params.dict['num_parallel_calls'],
                                                repeat=params.dict['repeat'],
                                                b_shuffle=params.dict['b_shuffle'],
                                                patches_per_class=params.dict['patches_per_class'],
                                                sigma_offset=params.dict['sigma_offset'],
                                                sigma_noise=params.dict['sigma_noise'],
                                                sigma_pos=params.dict['sigma_pos'],
                                                b_mirror=params.dict['b_mirror'],
                                                b_rotate=params.dict['b_rotate'],
                                                b_scale=params.dict['b_scale'],
                                                b_warp=params.dict['b_warp'],
                                                b_permute_labels=params.dict['b_permute_labels'],
                                                angle_max=params.dict['angle_max'],
                                                scale_factor=params.dict['scale_factor'],
                                                delta_max=params.dict['delta_max'],
                                                b_verbose=False)



    # Create the model (incorporating losses, optimizer, metrics)
    logging.info('Creating the model...')
    spec_model = model_fn.model_fn(spec_pipeline,
                                   input_metrics=None,
                                   b_training=b_training,
                                   channels=params.dict['channels'],
                                   channels_out=params.dict['channels_out'],
                                   batch_size=params.dict['size_batch'],
                                   b_dynamic_pos_mid=params.dict['b_dynamic_pos_mid'],
                                   b_dynamic_pos_end=params.dict['b_dynamic_pos_end'],
                                   non_local=non_local,
                                   non_local_num=1,
                                   attgate=attgate,
                                   filters=params.dict['filters'],
                                   dense_layers=params.dict['dense_layers'],
                                   alpha=params.dict['alpha'],
                                   dropout_rate=params.dict['rate_dropout'],
                                   rate_learning=params.dict['rate_learning'],
                                   beta1=params.dict['beta1'],
                                   beta2=params.dict['beta2'],
                                   epsilon=params.dict['epsilon'])


    # Train the actual model
    logging.info('Starting training for %i epoch(s)', params.dict['num_epochs'])
    training.sess_train(spec_pipeline, spec_model, params)


if __name__ == '__main__':
    # model
    model_dir = '/home/stage13_realshuffle_attgate_batch24_l2_repeat10_shuffle_eval_reminder_timout6000_K'

    # crucial parameters
    logging.info('Received the following input(s): %s', str(sys.argv))

    non_local = 'disable'
    if len(sys.argv) > 1:
        non_local = str(sys.argv[1])
    logging.info('Adding nonlocal block after %s', non_local)

    attgate = 'active'
    if len(sys.argv) > 2:
        attgate = istr(sys.argv[2])
    logging.info('Adding attention gate after %s', attgate)

    device = '3'
    if len(sys.argv) > 3:
        device = str(sys.argv[3])
    logging.info('Calculating on device %s', device)

    idx_dataset = 0
    if len(sys.argv) > 4:
        idx_dataset = int(sys.argv[4])
    logging.info('Using dataset %i', idx_dataset)

    main(model_dir, non_local=non_local, attgate=attgate, device=device, idx_dataset=idx_dataset)





