import tensorflow as tf
import math
import numpy as np

# TODO: some of the building blocks may be deprecated / not functional anymore -> check/fix


def block_pyramid(x, n_filters, dropout_rate, alpha, b_training, scope='block_pyramid'):

    with tf.variable_scope(scope):
        b0 = layer_conv3d(x, 256, [1, 1, 1])
        b1 = block_dense(x, 4, n_filters, dropout_rate=dropout_rate, alpha=alpha, b_training=b_training, rate_dil=[3, 3, 3])
        b2 = block_dense(x, 4, n_filters, dropout_rate=dropout_rate, alpha=alpha, b_training=b_training, rate_dil=[5, 5, 5])
        b3 = block_dense(x, 4, n_filters, dropout_rate=dropout_rate, alpha=alpha, b_training=b_training, rate_dil=[7, 7, 7])
        b4 = layer_global_average_pooling(x, 256, b_training=b_training)

        x.tfconcat([b0, b1, b2, b3, b4], axis=-1)
        x = layer_conv3d(x, 256, [1, 1, 1])

    return x


def block_dsp(x, max_branch, dropout_rate, alpha, b_training, scope='block_dsp'):
    """
     ESP/ASSP/ResUnit like block
     atm hardcoded
     uses full pre-activation
     """
    with tf.variable_scope(scope):
        # fetch shape for skip
        n_filters = x.get_shape().as_list()[-1]
        n_reduced = math.ceil(n_filters / max_branch)

        # full pre-activation 1x1 conv
        with tf.variable_scope('pre_act'):
            reduced = layer_batchnormalization(x, b_training=b_training)
            reduced = tf.nn.leaky_relu(reduced, alpha=alpha)
            reduced = layer_conv3d(reduced, n_reduced, [1, 1, 1])

        # dilated branches & hff sum concat
        with tf.variable_scope('dil_branches'):
            reduced = layer_batchnormalization(reduced, b_training=b_training)
            reduced = tf.nn.leaky_relu(reduced, alpha=alpha)

            d1 = None
            d2 = None
            # standard branch
            d0 = layer_conv3d(reduced, n_reduced, [3, 3, 3], dilation_rate=(1, 1, 1))
            d_concat = d0
            if max_branch >= 2:
                d1 = layer_conv3d(reduced, n_reduced, [3, 3, 3], dilation_rate=(2, 2, 2))
                d1 = tf.add(d0, d1)
                d_concat = tf.concat([d_concat, d1], axis=-1)
            if max_branch >= 3:
                d2 = layer_conv3d(reduced, n_reduced, [3, 3, 3], dilation_rate=(4, 4, 4))
                d2 = tf.add(d1, d2)
                d_concat = tf.concat([d_concat, d2], axis=-1)
            if max_branch >= 4:
                d3 = layer_conv3d(reduced, n_reduced, [3, 3, 3], dilation_rate=(8, 8, 8))
                d3 = tf.add(d2, d3)
                d_concat = tf.concat([d_concat, d3], axis=-1)

        # 1x1 conv
        with tf.variable_scope('add_skip'):
            d_concat = layer_batchnormalization(d_concat, b_training=b_training)
            d_concat = tf.nn.leaky_relu(d_concat, alpha=alpha)
            d_concat = layer_conv3d(d_concat, n_filters, [1, 1, 1])

            # + skip con
            x = tf.add(x, d_concat)

    return x


def block_dense(x, n_layers, growth_rate, dropout_rate, alpha, b_training, rate_dil=(1, 1, 1), scope='block_dense'):

    with tf.variable_scope(scope):
        layers_concat = list()
        layers_concat.append(x)

        with tf.variable_scope('layer0'):
            x = unit_bottleneck(x, growth_rate, dropout_rate=dropout_rate, alpha=alpha, b_training=b_training, rate_dil=rate_dil)
            layers_concat.append(x)

        for idx in range(n_layers - 1):
            with tf.variable_scope('layer_'+str(idx+1)):
                x = tf.concat(layers_concat, axis=-1)
                x = unit_bottleneck(x, growth_rate, dropout_rate=dropout_rate, alpha=alpha, b_training=b_training)
                layers_concat.append(x)

        x = tf.concat(layers_concat, axis=-1)

    return x


def unit_bottleneck(x, n_filters, dropout_rate, alpha, b_training, rate_dil=(1, 1, 1), scope='bottle'):
    """ growth-rate = n_filters are added per layer"""

    with tf.variable_scope(scope):
        # 1x1x1 conv followed by 3x3x3 conv
        with tf.variable_scope(scope + '_1'):
            x = layer_batchnormalization(x, b_training=b_training)
            x = tf.nn.leaky_relu(x, alpha=alpha)
            x = layer_conv3d(x, 4 * n_filters, [1, 1, 1])
            x = layer_dropout(x, dropout_rate, b_training)

        with tf.variable_scope(scope + '_2'):
            x = layer_batchnormalization(x, b_training=b_training)
            x = tf.nn.leaky_relu(x, alpha=alpha)
            x = layer_conv3d(x, n_filters, [3, 3, 3], dilation_rate=rate_dil)
            x = layer_dropout(x, dropout_rate=dropout_rate, b_training=b_training)

    return x


def unit_transition(x, n_filters, alpha, b_training, padding='same', scope='trans'):

    with tf.variable_scope(scope):
        x = layer_batchnormalization(x, b_training=b_training)
        x = tf.nn.leaky_relu(x, alpha=alpha)
        x = layer_conv3d(x, n_filters, [3, 3, 3], strides=(2, 2, 2), padding=padding)  # TODO: adjust filters correctly
        # paper used average pooling with preceding dropout

    return x


def unit_transition_up(x, n_filters, alpha, b_training, padding='same', scope='trans_up'):

    with tf.variable_scope(scope):
        x = layer_batchnormalization(x, b_training)
        x = tf.nn.leaky_relu(x, alpha=alpha)
        x = layer_conv3d_transpose(x, n_filters, [3, 3, 3], strides=(2, 2, 2), padding=padding)

    return x


def layer_batchnormalization(x, b_training):

    return tf.layers.batch_normalization(x, training=b_training)


def layer_conv3d(x, n_filters, kernel_size, strides=(1, 1, 1), padding='same', dilation_rate=(1, 1, 1), use_bias=False):

    return tf.layers.conv3d(x,
                            filters=n_filters,
                            kernel_size=kernel_size,
                            strides=strides,
                            padding=padding,
                            dilation_rate=dilation_rate,
                            use_bias=use_bias)


def layer_conv3d_transpose(x, n_filters, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same', use_bias=False, name=None):

    return tf.layers.conv3d_transpose(x,
                                      filters=n_filters,
                                      kernel_size=kernel_size,
                                      strides=strides,
                                      padding=padding,
                                      use_bias=use_bias,
                                      name=name)


def layer_conv3d_pre_ac(x, b_training, alpha, n_filters, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', dilation_rate=(1, 1, 1), use_bias=False, scope='conv3d_pre_ac'):

    with tf.variable_scope(scope):
        x = layer_batchnormalization(x, b_training)
        x = tf.nn.leaky_relu(x, alpha=alpha)
        x = layer_conv3d(x, n_filters=n_filters, kernel_size=kernel_size, strides=strides, padding=padding, dilation_rate=dilation_rate, use_bias=use_bias)

    return x


def layer_dropout(x, dropout_rate, b_training, name=None):

    return tf.layers.dropout(x, training=b_training, rate=dropout_rate, name=name)


def layer_softmax(x):

    return tf.nn.softmax(x, axis=-1)


def layer_global_average_pooling(x, n_filters, b_training, scope='glob_avg'):

    with tf.variable_scope(scope):
        temp_shape = x.get_shape()
        x = tf.reduce_mean(x, axis=[1, 2, 3], keepdims=True)
        x = layer_conv3d(x, n_filters, [1, 1, 1])
        x = layer_batchnormalization(x, b_training=b_training)
        x = tf.tile(x, [temp_shape[1], temp_shape[2], temp_shape[3]])

    return x


def unit_dynamic_conv(inputs, inputs_gen, batch_size, channels_out, filter_size=(1, 1, 1), strides=(1, 1, 1, 1, 1), alpha=0.2, padding='SAME', dilations=(1, 1, 1, 1, 1), scope='unit_dyn_conv'):
    """
    Create filters dynamically based on inputs_gen and apply them to inputs
    :param inputs:
    :param inputs_gen:
    :param channels_out:
    :param filter_size:
    :param strides:
    :param alpha:
    :param padding:
    :param dilations:
    :param scope:
    :return:
    """
    # TODO: decide/test whether to use dynamic conv or dynamic filtering
    with tf.variable_scope(scope):

        samples = batch_size
        channels_in = inputs.get_shape()[-1]
        # inputs_gen = tf.Print(inputs_gen, [tf.shape(inputs_gen)], 'fetched positions: ')
        split_input_gen = tf.split(inputs_gen, samples, axis=0)

        # integrate reduce layer (so layer count isn't too high
        # inputs = layer_conv3d(inputs, channels_in, [1, 1, 1])

        print(inputs_gen, filter_size)

        filter_list = list()
        for input_vars in split_input_gen:
            with tf.variable_scope('dyn_filter_gen', reuse=tf.AUTO_REUSE):
                # input_vars = tf.Print(input_vars, [input_vars], 'input_vars for single sample: ')
                filter_gen = tf.layers.dense(input_vars, 64, name='gen0')
                filter_gen = tf.nn.leaky_relu(filter_gen, alpha=alpha, name='relu0')
                filter_gen = tf.layers.dense(filter_gen, 128, name='gen1')
                filter_gen = tf.nn.leaky_relu(filter_gen, alpha=alpha, name='relu1')
                filter_gen = tf.layers.dense(filter_gen, channels_in*channels_out*np.prod(filter_size), name='gen2')
                filter_gen = tf.reshape(filter_gen, [*filter_size, channels_in, channels_out], name='reshape0')
                filter_list.append(filter_gen)

        filters_gen = tf.stack(filter_list, axis=0)

        filtered_input = layer_dynamic_conv(inputs, batch_size, filters_gen, strides=strides, padding=padding, dilations=dilations)

    return filtered_input


def layer_dynamic_conv(inputs, batch_size, filters, strides=(1, 1, 1, 1, 1), padding='SAME', dilations=(1, 1, 1, 1, 1), scope='dyn_conv'):
    """

    :param inputs: input batch-wise convolutions are applied to
    :param filters: filters that are split according to batch size and applied to each corresponding input
    :param kernel_size:
    :param strides:
    :param padding:
    :param dilations:
    :param scope:
    :return:

    source: https://github.com/tensorflow/tensorflow/issues/16831
    """

    with tf.variable_scope(scope):
        # from gen_input some filter weights are created and a convolution with those onto x is performed

        samples = batch_size  # inputs.get_shape().as_list()[0]
        split_inputs = tf.split(inputs, samples, axis=0)
        split_filters = tf.unstack(filters, samples, axis=0)

        output_list = list()
        for split_input, split_filter in zip(split_inputs, split_filters):
            output_list.append(tf.nn.conv3d(split_input,
                                            split_filter,
                                            strides=strides,
                                            padding=padding,
                                            dilations=dilations))

        output = tf.concat(output_list, axis=0)

    return output



def nonlocalblock(input_x, b_training, group_num=1, half_channels=True, use_maxpool=True, is_bn=False, mode='embedded', scope='nonlocalblock'):
    """
            implement a nonlocal block to the model
            :param inputs: input tensor
            :param b_training: is training or not
            :param group_num: cut the input data in slices, in case the GPU memory usage not enough for performing non-local OP on 3d data
            :param half_channels: use bottleneck design to reduce the channels to half
            :param use_maxpool: use subsampling trick to reduce the computation
            :param is_bn: bias should be used in the last scope 'w', or not  .
            :param mode: function f that be used, in ['gaussian', 'embedded', 'dot', 'concatenate'] 
            :return: return the tensor which has the same shape as the input tensor

            """
    if mode not in ['gaussian', 'embedded', 'dot']:
        raise ValueError('`mode` must be one of `gaussian`, `embedded` or `dot`')

    batchsize, height, width, depth, in_channels = input_x.get_shape().as_list()

    if half_channels:   #bottleneck to reduce computation
        inner_channels = int(in_channels / 2)
        if in_channels < 1:
            inner_channels = 1
    else:
        inner_channels = in_channels

    cache_in = input_x
    
    assert (height % group_num == 0)
    if group_num > 1:   #group_num>1 ---> "cut" the data
        cache_in = tf.reshape(cache_in, [batchsize*group_num, int(height/group_num), width, depth, in_channels])
    assert ((height/group_num) % 2 == 0)

    if use_maxpool:     #subsampling trick
        max_pool = tf.nn.max_pool3d(cache_in, [1, 2, 2, 2, 1], [1, 2, 2, 2, 1], padding='VALID')
    else:
        max_pool = cache_in

    if mode == 'embedded':
        with tf.variable_scope('embedded_theta', reuse=tf.AUTO_REUSE):
            theta = layer_conv3d(cache_in, inner_channels, [1, 1, 1])

        with tf.variable_scope('embedded_phi', reuse=tf.AUTO_REUSE):
            phi = layer_conv3d(max_pool, inner_channels, [1,1,1])
            #phi = tf.nn.max_pool3d(phi, [1, 2, 2, 2, 1], [1, 2, 2, 2, 1], padding='VALID')

        with tf.variable_scope('embedded_g', reuse=tf.AUTO_REUSE):
            g = layer_conv3d(max_pool, inner_channels, [1,1,1])
            #g = tf.nn.max_pool3d(g, [1, 2, 2, 2, 1], [1, 2, 2, 2, 1], padding='VALID')

        theta = tf.reshape(theta, [batchsize*group_num, -1, inner_channels])#e.g. shape in 784x512
        phi = tf.reshape(phi, [batchsize*group_num, -1, inner_channels])
        phi = tf.transpose(phi, [0, 2, 1])  #e.g. shape in 512x784
        g = tf.reshape(g, [batchsize*group_num, -1, inner_channels])
        g = tf.transpose(g, [0, 2, 1]) #e.g.shape in 512x784
        f = tf.matmul(theta, phi) #e.g.shape in 784_1x784_2
        f = tf.nn.softmax(f, axis=2) #e.g.shape in 784_1x784_2
        f = tf.transpose(f, [0, 2, 1]) #e.g.shape in 784_2x784_1

    elif mode == 'gaussian':
        with tf.variable_scope('gaussian_theta', reuse=tf.AUTO_REUSE):
            theta = tf.reshape(cache_in, [batchsize * group_num, -1, in_channels])

        with tf.variable_scope('gaussian_phi', reuse=tf.AUTO_REUSE):
            phi = tf.reshape(max_pool, [batchsize * group_num, -1, in_channels])
            phi = tf.transpose(phi, [0, 2, 1])  # e.g. shape in 512x784
            #phi = tf.nn.max_pool3d(phi, [1, 2, 2, 2, 1], [1, 2, 2, 2, 1], padding='VALID')

        with tf.variable_scope('gaussian_g', reuse=tf.AUTO_REUSE):
            g = layer_conv3d(max_pool, inner_channels, [1,1,1])
            #g = tf.nn.max_pool3d(g, [1, 2, 2, 2, 1], [1, 2, 2, 2, 1], padding='VALID')

        g = tf.reshape(g, [batchsize*group_num, -1, inner_channels])
        g = tf.transpose(g, [0, 2, 1]) #e.g.shape in 512x784
        f = tf.matmul(theta, phi) #e.g. shape in 784_1x784_2
        f = tf.nn.softmax(f, axis=2) #e.g. shape in 784_1x784_2
        f = tf.transpose(f, [0, 2, 1]) #e.g. shape in 784_2x784_1

    else:   #dot mode
        with tf.variable_scope('dot_theta', reuse=tf.AUTO_REUSE):
            theta = layer_conv3d(cache_in, inner_channels, [1, 1, 1])

        with tf.variable_scope('dot_phi', reuse=tf.AUTO_REUSE):
            phi = layer_conv3d(max_pool, inner_channels, [1,1,1])
            #phi = tf.nn.max_pool3d(phi, [1, 2, 2, 2, 1], [1, 2, 2, 2, 1], padding='VALID')

        with tf.variable_scope('dot_g', reuse=tf.AUTO_REUSE):
            g = layer_conv3d(max_pool, inner_channels, [1,1,1])
            #g = tf.nn.max_pool3d(g, [1, 2, 2, 2, 1], [1, 2, 2, 2, 1], padding='VALID')

        theta = tf.reshape(theta, [batchsize*group_num, -1, inner_channels])#784x512
        phi = tf.reshape(phi, [batchsize*group_num, -1, inner_channels])
        phi = tf.transpose(phi, [0, 2, 1])  # 512x784
        g = tf.reshape(g, [batchsize*group_num, -1, inner_channels])
        g = tf.transpose(g, [0, 2, 1]) #512x784
        f = tf.matmul(theta, phi) #784_1x784_2
        size = tf.shape(f)
        f = f / tf.cast(size[-1], tf.float32)
        f = tf.transpose(f, [0, 2, 1]) #784_2x784_1

    y = tf.matmul(g, f) #e.g. shape in 512x784_1
    y = tf.transpose(y, [0, 2, 1]) #e.g. shape in 784_1x512
    y = tf.reshape(y, [batchsize*group_num, int(height/group_num), width, depth, inner_channels])

    with tf.variable_scope('w', reuse=tf.AUTO_REUSE):
        w_y = layer_conv3d(y, in_channels, [1,1,1])
        if is_bn:
            w_y = layer_batchnormalization(w_y, b_training=b_training)
            # resnet connection
    z = tf.add(cache_in, w_y)
    z = tf.reshape(z, [batchsize, height, width, depth, in_channels])
        #output_list.append(z)
    #z_sum = tf.add_n(output_list)

    return z


def attgate(input_x, gate_x, half_channels=True, scope='attgate'):
    """
            implement attention gates to the model
            :param inputs: input tensor
            :param gate_x: gate signal from downside
            :param half_channels: to reduce computation cut the channels to half
            :return: return the tensor at the skip connection in the unet

            """

    batchsize, height, width, depth, in_channels = input_x.get_shape().as_list()
    b,h,w,d,gc = gate_x.get_shape().as_list()
    assert (batchsize == b)
    if half_channels:
        inner_channels = int(in_channels / 2)
        if in_channels < 1:
            inner_channels = 1
    else:
        inner_channels = in_channels

    cache_in = input_x

    with tf.variable_scope('gate_theta', reuse=tf.AUTO_REUSE):
        theta = layer_conv3d(cache_in, inner_channels, [2, 2, 2], strides=(2, 2, 2), padding='VALID')

        #self.phi = conv_nd(in_channels=self.gating_channels, out_channels=self.inter_channels,
         #                  kernel_size=1, stride=1, padding=0, bias=True)
    with tf.variable_scope('geta_g', reuse=tf.AUTO_REUSE):
        g = layer_conv3d(gate_x, inner_channels, [1, 1, 1], use_bias=True)
        g = layer_conv3d_transpose(g, inner_channels, [3, 3, 3], strides=(1, 1, 1), padding='same')

    f = tf.nn.relu(theta + g)
    f = layer_conv3d(f, 1, [1, 1, 1], use_bias=True) #shape=(24, 4, 4, 4, 1)

    sig_f = tf.sigmoid(f) #shape=(24, 4, 4, 4, 1)
    sig_f = layer_conv3d_transpose(sig_f, 1, [3, 3, 3], strides=(2, 2, 2), padding='same') #shape=(24, 8, 8, 8, 1)
    sig_f = tf.tile(sig_f, [1, 1, 1, 1, in_channels])
    y = sig_f * cache_in
    w_y = layer_conv3d(y, in_channels, [1, 1, 1], strides=(1, 1, 1))

    return w_y