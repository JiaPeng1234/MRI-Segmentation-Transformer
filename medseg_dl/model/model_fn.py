import tensorflow as tf
from medseg_dl.model import metrics
from medseg_dl.model import losses
from medseg_dl.utils import utils_patching
import medseg_dl.model.layers as cstm_layers
import os
import collections

# TODO: instances that need predictions/probabilities instead of logits should be fed this way directly


def model_fn(input,
             input_metrics,
             b_training,
             channels,
             channels_out,
             batch_size,
             b_dynamic_pos_mid=False,
             b_dynamic_pos_end=False,
             non_local='disable',
             non_local_num=1,
             attgate='disable',
             filters=32,
             dense_layers=4,
             alpha=0.0,
             dropout_rate=0.0,
             rate_learning=0.001,
             beta1=0.9,
             beta2=0.999,
             epsilon=1e-8,
             b_verbose=False):
    """Model function defining the graph operations. """

    """ prediction """
    with tf.variable_scope('model'):

        """ network """
        with tf.variable_scope('network'):
            logits = _build_model(input['images'],
                                  b_training,
                                  channels,
                                  channels_out,
                                  batch_size,
                                  b_dynamic_pos_mid,
                                  b_dynamic_pos_end,
                                  non_local,
                                  non_local_num,
                                  attgate,
                                  input['positions'],
                                  filters,
                                  dense_layers,
                                  alpha,
                                  dropout_rate)

            if b_verbose:
                logits = tf.Print(logits, [input['positions']], 'used patch positions: ', summarize=50)
                logits = tf.Print(logits, [tf.shape(logits)], 'prediction shape: ', summarize=5)
            probs = tf.nn.softmax(logits)
            if not b_training:
                """ evaluation by patch aggregation """
                # define aggregation variable that holds the image probs
                agg_probs = tf.get_variable('agg_probs',
                                            shape=[input['n_tiles'], *input['shape_output'], channels_out],
                                            dtype=tf.float32,
                                            initializer=tf.zeros_initializer,
                                            trainable=False,
                                            collections=[tf.GraphKeys.LOCAL_VARIABLES],
                                            use_resource=True)

                batch_count = tf.get_variable('batch_count',
                                              shape=[1],
                                              dtype=tf.int32,
                                              initializer=tf.zeros_initializer,
                                              trainable=False,
                                              collections=[tf.GraphKeys.LOCAL_VARIABLES],
                                              use_resource=True)

                recombined_probs = tf.get_variable('recombined_probs',
                                                   shape=[1, *input['shape_image'], channels_out],
                                                   dtype=tf.float32,
                                                   initializer=tf.zeros_initializer,
                                                   trainable=False,
                                                   collections=[tf.GraphKeys.LOCAL_VARIABLES],
                                                   use_resource=True)
                recombined_probs_value = recombined_probs.read_value()  # tensor to value / may also happen automatically

                # make initializer
                agg_probs_init_op = tf.variables_initializer(tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope=os.path.join(tf.get_default_graph().get_name_scope())))

                # aggregate each batch run
                op_batch = batch_count.assign_add([tf.shape(probs)[0]])
                # update_op_agg_probs = tf.group(*[op_batch, op_agg_probs])

                with tf.control_dependencies([op_batch]):  # make sure batch is updated if aggregation is performed (at least for test cases)
                    if b_verbose:
                        probs = tf.Print(probs, [batch_count.read_value()], 'batch_count: ')
                        probs = tf.Print(probs, [tf.range(batch_count[0] - tf.shape(probs)[0], batch_count[0])], 'using range: ', summarize=50)

                    # update part of agg_probs with current prediction
                    # atm last dummy batch part is discarded
                    # agg_probs = tf.scatter_update(agg_probs, tf.range(batch_count[0] - tf.shape(probs)[0], batch_count[0]), probs)
                    agg_probs = tf.cond(tf.squeeze(tf.greater_equal(batch_count, input['n_tiles'])),
                            true_fn=lambda: tf.scatter_update(agg_probs,
                                                              tf.range(batch_count[0] - tf.shape(probs)[0], input['n_tiles']),
                                                              probs[:tf.shape(probs)[0]+input['n_tiles']-batch_count[0], ...]),
                            false_fn=lambda: tf.scatter_update(agg_probs, tf.range(batch_count[0] - tf.shape(probs)[0], batch_count[0]), probs))

                    # perform final conversion
                    # Note: this should only be executed after a whole image has been aggregated as batch patches
                    # wrap assignment op in a cond so it isn't executed all the time
                    # recombined_probs_op = recombined_probs.assign(input_fn.batch_to_space(agg_probs, input['shape_padded_label'], input['shape_image'], channels_out))
                    recombined_probs_op = tf.cond(tf.squeeze(tf.greater_equal(batch_count, input['n_tiles'])),
                                                  true_fn=lambda: recombined_probs.assign(
                                                      utils_patching.batch_to_space(agg_probs, input['tiles'], input['shape_padded_label'], input['shape_image'], channels_out, b_verbose=b_verbose)),
                                                  false_fn=lambda: recombined_probs)  # dummy assignment
                    # recombined_probs.assign(recombined_probs) failed as dummy assignment. why?

            else:
                agg_probs = None
                agg_probs_init_op = None
                recombined_probs_value = None
                recombined_probs_op = None

        """ training """
        if b_training:
            """ losses """
            with tf.variable_scope('losses'):
                loss = losses.soft_jaccard(labels=input['labels'], probs=probs)

            with tf.variable_scope('model/summary/'):
                # loss
                tf.summary.scalar('loss_soft_jaccard', loss, collections=['summaries_train'], family='loss')
                summary_op_train = tf.summary.merge_all(key='summaries_train')

            """ optimizer """
            with tf.variable_scope('optimizer'):
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # BN and co. need it
                with tf.control_dependencies([loss, *update_ops]):
                    global_step = tf.train.get_or_create_global_step()

                    # the following three lines are equivalent to tf.train.<Optimizer>.minimize(loss)
                    optimizer = tf.train.AdamOptimizer(learning_rate=rate_learning, beta1=beta1, beta2=beta2, epsilon=epsilon)
                    gradients = optimizer.compute_gradients(loss, var_list=tf.trainable_variables())
                    train_op = optimizer.apply_gradients(gradients, global_step=global_step)

        else:
            loss = None
            summary_op_train = None
            train_op = None

        """ metrics """
        with tf.variable_scope('metrics'):
            if b_training:
                init_op_metrics, update_op_metrics, metrics_values = metrics.metrics_fn(input['labels'], probs, channels_out=channels_out)
            else:
                print('labels', input['labels'])
                print('recombined_probs', recombined_probs_value)
                init_op_metrics, update_op_metrics, metrics_values = metrics.metrics_fn(input_metrics['labels'], recombined_probs_value, channels_out=channels_out)

        with tf.variable_scope('model/summary/'):
            for k, v in metrics_values.items():
                tf.summary.scalar(k, v, collections=['summaries_metrics'], family='metrics')
            summary_op_metrics = tf.summary.merge_all(key='summaries_metrics')

    # generate the model spec
    spec_model = collections.defaultdict(None)
    spec_model['logits'] = logits
    spec_model['probs'] = probs

    # training
    spec_model['loss'] = loss
    spec_model['train_op'] = train_op
    spec_model['summary_op_train'] = summary_op_train

    # evaluation
    spec_model['agg_probs'] = agg_probs
    spec_model['agg_probs_init_op'] = agg_probs_init_op
    spec_model['recombined_probs_value'] = recombined_probs_value
    spec_model['recombined_probs_op'] = recombined_probs_op

    # metrics
    spec_model['init_op_metrics'] = init_op_metrics
    spec_model['update_op_metrics'] = update_op_metrics
    spec_model['metrics_values'] = metrics_values
    spec_model['summary_op_metrics'] = summary_op_metrics

    return spec_model




def _build_model(l0, b_training, channels_in, channels_out, batch_size, b_dynamic_pos_mid, b_dynamic_pos_end, non_local='disable', non_local_num=1, attgate='disable', positions=None, filters=32, dense_layers=4, alpha=0.0, dropout_rate=0.0):
    # TODO: dense_layer property is still hardcoded in _build_model()
    # all blocks/units are in general using BN/Relu at the start so normal conv is sufficient
    # Note: valid padding -> output is reduced -> this has to be reflected in input_fn
    if non_local not in ['input', 'l0', 'l1', 'l2', 'bottleneck', 'disable']:
        raise ValueError('`nonlocal` must be one of `input`, `l0`, `l1`, `l2`, `bottleneck` or `disable`')
    if attgate not in ['active', 'disable']:
        raise ValueError('`attgate` must be one of `input`, `l0`, `l1`, `l2`, `bottleneck` or `disable`')
    assert not((non_local!='disable') and (attgate!='disable')) #cant be used at the same time
    with tf.variable_scope('encoder'):
        if non_local=='input':
            with tf.variable_scope('no_local_input'):
            #insert at the beginning stage
                l0 = cstm_layers.nonlocalblock(l0, b_training, scope='no_local_input')

        with tf.variable_scope('l0'):  # in 64x64x64
            l0 = cstm_layers.layer_conv3d(l0, 32, [5, 5, 5], strides=(1, 1, 1), padding='valid')  # 64x64x64 -> 60x60x60
            l0 = cstm_layers.layer_conv3d_pre_ac(l0, b_training, alpha, 32, kernel_size=(3, 3, 3), padding='valid')  # 60x60x60 -> 58x58x58

        if non_local=='l0':
            with tf.variable_scope('no_local_l0'):
                l0 = cstm_layers.nonlocalblock(l0, b_training, scope='no_local_l0')

        with tf.variable_scope('l1'):  # in 58x58x58
            l1 = cstm_layers.unit_transition(l0, 64, alpha=alpha, b_training=b_training, scope='trans0', padding='valid')  # 58x58x58 -> 56x56x56/2 -> 28x28x28
            l1 = cstm_layers.layer_conv3d_pre_ac(l1, b_training, alpha, 64, kernel_size=(3, 3, 3), padding='valid')  # 28x28x28 -> 26x26x26
            # l1 = cstm_layers.block_dense(l1, 4, 12, dropout_rate=dropout_rate, alpha=alpha, b_training=b_training, scope='dense0')  # 16 + 4*12 = 64

        if non_local=='l1':
            for idx in range(non_local_num):
                with tf.variable_scope('no_local_l1_{}'.format(idx)):
                    l1 = cstm_layers.nonlocalblock(l1, b_training, scope='no_local_l1_{}'.format(idx))

            ####l1 = cstm_layers.block_dsp(l1, max_branch=3, dropout_rate=dropout_rate, alpha=alpha, b_training=b_training, scope='dsp0')  # same

        with tf.variable_scope('l2'):  # in 26x26x26
            l2 = cstm_layers.unit_transition(l1, 128, alpha=alpha, b_training=b_training, scope='trans1', padding='valid')  # 26x26x26 -> 24x24x24/2 -> 12x12x12
            l2 = cstm_layers.layer_conv3d_pre_ac(l2, b_training, alpha, 128, kernel_size=(3, 3, 3), padding='valid')  # 12x12x12 -> 10x10x10
            # l2 = cstm_layers.block_dense(l2, 4, 12, dropout_rate=dropout_rate, alpha=alpha, b_training=b_training, scope='dense1')  # 80 + 4*12 = 64

        if non_local=='l2':
            for idx in range(non_local_num):
                with tf.variable_scope('no_local_l2_{}'.format(idx)):
                    l2 = cstm_layers.nonlocalblock(l2, b_training, scope='no_local_l2_{}'.format(idx))

            ####l2 = cstm_layers.block_dsp(l2, max_branch=2, dropout_rate=dropout_rate, alpha=alpha, b_training=b_training, scope='dsp1')  # same

    with tf.variable_scope('bottleneck'):  # in 10x10x10
        l3 = cstm_layers.unit_transition(l2, 256, alpha=alpha, b_training=b_training, scope='trans2', padding='valid')  # 10x10x10 -> 8x8x8/2 -> 4x4x4
        l3 = cstm_layers.layer_conv3d(l3, 256, [1, 1, 1])  # 4x4x4,gating

    if non_local=='bottleneck':
        for idx in range(non_local_num):
            with tf.variable_scope('no_local_bottleneck_{}'.format(idx)):
                l3 = cstm_layers.nonlocalblock(l3, b_training, scope='no_local_bottleneck_{}'.format(idx))
        ####if b_dynamic_pos_mid:
            ####l3 = cstm_layers.unit_dynamic_conv(l3, positions, batch_size, 256, filter_size=(1, 1, 1), strides=(1, 1, 1, 1, 1), alpha=alpha, padding='SAME', dilations=(1, 1, 1, 1, 1), scope='unit_dyn_conv_mid')
        # l3 = cstm_layers.block_dense(l3, 4, 12, dropout_rate=dropout_rate, alpha=alpha, b_training=b_training, scope='dense2')  # 160 + 4*12 = 256

    with tf.variable_scope('decoder'):
        with tf.variable_scope('l2_up'):  # in 4x4x4
            #l2_att = cstm_layers.attgate(l2[:, 1:9, 1:9, 1:9, :], l3) #8x8x8
            l2_up = cstm_layers.unit_transition_up(l3, 128, alpha=alpha, b_training=b_training, scope='trans_up0')  # 4x4x4 -> 8x8x8
            if attgate == 'disable':
                l2_up = tf.concat([l2[:, 1:9, 1:9, 1:9, :], l2_up], axis=-1)  # needs 8x8x8
            else:
                l2_att = cstm_layers.attgate(l2[:, 1:9, 1:9, 1:9, :], l3)  # 8x8x8
                l2_up = tf.concat([l2_att, l2_up], axis=-1)
            l2_up = cstm_layers.layer_conv3d_pre_ac(l2_up, b_training, alpha, 128, kernel_size=(3, 3, 3), padding='valid')  # 8x8x8 -> 6x6x6


            ####l2_up = cstm_layers.block_dsp(l2_up, max_branch=2, dropout_rate=dropout_rate, alpha=alpha, b_training=b_training, scope='dsp_up1')  # same

        with tf.variable_scope('l1_up'):  # in 6x6x6
            #l1_att = cstm_layers.attgate(l1[:, 7:19, 7:19, 7:19, :], l2_up) #12x12x12
            l1_up = cstm_layers.unit_transition_up(l2_up, 64, alpha=alpha, b_training=b_training, scope='trans_up1')  # 6x6x6 -> 12x12x12
            if attgate == 'disable':
                l1_up = tf.concat([l1[:, 7:19, 7:19, 7:19, :], l1_up], axis=-1)  # needs 12x12x12
            else:
                l1_att = cstm_layers.attgate(l1[:, 7:19, 7:19, 7:19, :], l2_up)  # 12x12x12
                l1_up = tf.concat([l1_att, l1_up], axis=-1)
            l1_up = cstm_layers.layer_conv3d_pre_ac(l1_up, b_training, alpha, 64, kernel_size=(3, 3, 3), padding='valid')  # 12x12x12 -> 10x10x10


            ####l1_up = cstm_layers.block_dsp(l1_up, max_branch=3, dropout_rate=dropout_rate, alpha=alpha, b_training=b_training, scope='dsp_up2')  # same

        with tf.variable_scope('l0_up'):  # in 10x10x10
            #l0_att = cstm_layers.attgate(l0[:, 19:39, 19:39, 19:39, :], l1_up)  # 20x20x20
            l0_up = cstm_layers.unit_transition_up(l1_up, 32, alpha=alpha, b_training=b_training, scope='trans_up2')  # 10x10x10 -> 20x20x20
            if attgate == 'disable':
                l0_up = tf.concat([l0[:, 19:39, 19:39, 19:39, :], l0_up], axis=-1)  # needs 20x20x20
            else:
                l0_att = cstm_layers.attgate(l0[:, 19:39, 19:39, 19:39, :], l1_up)  # 20x20x20
                l0_up = tf.concat([l0_att, l0_up], axis=-1)
            l0_up = cstm_layers.layer_conv3d_pre_ac(l0_up, b_training, alpha, 32, kernel_size=(3, 3, 3), padding='valid', scope='conv3d_pre_ac1')  # 20x20x20 -> 18x18x18
            ####if b_dynamic_pos_end:
                ####l0_up = cstm_layers.unit_dynamic_conv(l0_up, positions, batch_size, 32, filter_size=(1, 1, 1), strides=(1, 1, 1, 1, 1), alpha=alpha, padding='SAME', dilations=(1, 1, 1, 1, 1), scope='unit_dyn_conv_end')
            l0_up = cstm_layers.layer_conv3d_pre_ac(l0_up, b_training, alpha, 32, kernel_size=(3, 3, 3), padding='valid', scope='conv3d_pre_ac2')  # 18x18x18 -> 16x16x16
            l0_up = cstm_layers.layer_conv3d_pre_ac(l0_up, b_training, alpha, channels_out, kernel_size=(1, 1, 1), scope='conv3d_pre_ac3')  # 16x16x16

    return l0_up
