import os
import logging
import tensorflow as tf
from medseg_dl import parameters
from medseg_dl.utils import utils_data, utils_misc
from medseg_dl.model import input_fn
from medseg_dl.model import model_fn
from medseg_dl.model import evaluation


def main(dir_model, non_local = 'disable', attgate = 'disable', device = None):

    # Since this is the evaluation script all ops should switch to prediction
    b_training = False

    # Load parameters from model file
    file_params = os.path.join(dir_model, 'params.yaml')
    assert os.path.isfile(file_params)  # file has to exist!
    params = parameters.Params(path_yaml=file_params)

    # Set logger
    utils_misc.set_logger(os.path.join(params.dict['dir_model'], 'eval.log'), params.dict['log_level'])

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
    # Note: one can either do evaluation on small little patches or the whole image -> has to be reflected in the model
    logging.info('Creating the pipeline...')
    logging.info(f'Evaluating {filenames_eval}')
    spec_pipeline = input_fn.gen_pipeline_eval_patch(filenames=filenames_eval,
                                                     shape_image=params.dict['shape_image_eval'],
                                                     shape_input=params.dict['shape_input'],
                                                     shape_output=params.dict['shape_output'],
                                                     size_batch=params.dict['size_batch_eval'],
                                                     channels_out=params.dict['channels_out'],
                                                     size_buffer=params.dict['size_buffer'],
                                                     num_parallel_calls=params.dict['num_parallel_calls'],
                                                     b_with_labels=params.dict['b_eval_labels_patch'],
                                                     b_verbose=True)

    spec_pipeline_metrics = input_fn.gen_pipeline_eval_image(filenames=filenames_eval,
                                                             shape_image=params.dict['shape_image_eval'],
                                                             channels_out=params.dict['channels_out'],
                                                             size_batch=1,
                                                             size_buffer=params.dict['size_buffer'],
                                                             num_parallel_calls=params.dict['num_parallel_calls'],
                                                             b_with_labels=params.dict['b_eval_labels_image'],
                                                             b_verbose=True)

    # Create the model
    logging.info('Creating the model...')
    spec_model = model_fn.model_fn(spec_pipeline,
                                   spec_pipeline_metrics,
                                   b_training=b_training,
                                   channels=params.dict['channels'],
                                   channels_out=params.dict['channels_out'],
                                   batch_size=params.dict['size_batch_eval'],
                                   b_dynamic_pos_mid=params.dict['b_dynamic_pos_mid'],
                                   b_dynamic_pos_end=params.dict['b_dynamic_pos_end'],
                                   non_local=non_local,
                                   non_local_num=1,
                                   attgate=attgate,
                                   filters=params.dict['filters'],
                                   dense_layers=params.dict['dense_layers'],
                                   alpha=params.dict['alpha'],
                                   dropout_rate=params.dict['rate_dropout'],
                                   b_verbose=False)

    # Evaluate a saved model
    logging.info('Starting evaluation...')
    evaluation.sess_eval(spec_pipeline, spec_pipeline_metrics, spec_model, params, filenames_eval)


if __name__ == '__main__':

    model_dir_base = '/home/stage13_realshuffle_attgate_batch24_l2_repeat10_shuffle_eval_reminder_timout6000_K'

    run = 'catalog_in_model_dir_base'
    model_dir = os.path.join(model_dir_base, run)
    device = '2'
    non_local = 'disable'
    attgate = 'active'
    main(model_dir, non_local=non_local, attgate=attgate, device=device)

