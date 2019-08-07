import tensorflow as tf
from medio import convert_tf
import numpy as np
from medseg_dl.utils import utils_patching, utils_augmentation


def gen_pipeline_train(filenames,
                       shape_image,
                       shape_input,
                       shape_output,
                       channels_out,
                       size_batch,
                       size_buffer=1,
                       num_parallel_calls=1,
                       repeat=1,
                       b_shuffle=True,
                       patches_per_class=(3, 1, 1, 1, 1),
                       sigma_offset=0.1,
                       sigma_noise=0.05,
                       sigma_pos=0.05,
                       b_mirror=False,
                       b_rotate=False,
                       b_scale=False,
                       b_warp=False,
                       b_permute_labels=False,
                       angle_max=0,
                       scale_factor=0,
                       delta_max=0,
                       b_verbose=False):
    """ generates the tf data pipeline
        makes use of tfrecords
    """

    # generate values needed during mapping
    paddings_image = [int((a - b) / 2) for a, b in zip(shape_input, shape_output)]
    shape_padded_image = [a + 2 * b for a, b in zip(shape_image, paddings_image)]

    def _map_data(*args):  # process passed tensors (care: nested structure)

        # order input
        n_images = len(filenames[0])
        n_labels = len(filenames[1])

        imgs = list(args[0:n_images])
        labs = list(args[n_images:n_images+n_labels])  # allows for further labels
        # Note: current processing just considers labs[0]

        # process imgs
        for idx in range(n_images):
            # read tf records
            imgs[idx] = convert_tf.parse_function(imgs[idx])

            # cast images
            imgs[idx] = tf.cast(imgs[idx], tf.float32)
            # b = tf.cast(b, tf.float32)

            # image normalization:
            tmp_mean, tmp_var = tf.nn.moments(imgs[idx], axes=[0, 1, 2])
            imgs[idx] = tf.divide(imgs[idx] - tmp_mean,
                                  tf.maximum(tf.sqrt(tmp_var),
                                             tf.divide(1.0, tf.sqrt(tf.cast(tf.size(imgs[idx]), dtype=tf.float32)))))

            """ padding """
            # pad so a label prediction would be possible on full image (makes it more equal to eval case)
            imgs[idx] = tf.pad(imgs[idx], [(x, x) for x in paddings_image])

        for idx in range(n_labels):
            labs[idx] = convert_tf.parse_function(labs[idx])

            # process labels
            labs[idx] = tf.clip_by_value(tf.cast(labs[idx], dtype=tf.int32), 0, channels_out - 1)

            labs[idx] = tf.pad(labs[idx], [(x, x) for x in paddings_image])

        # check that the correct data / shapes are provided - remember it's all a graph, so use special tf.Print magic
        if b_verbose:
            imgs[0] = tf.Print(imgs[0], [tf.shape(imgs[0]), tf.shape(labs[0])], 'fetched data shapes: ')

        # ensure correct dims if using whole images
        # a = tf.slice(a, [0, 0, 0], shape_input)
        # b = tf.slice(b, [0, 0, 0], shape_input)
        # c = tf.slice(c, [0, 0, 0], shape_input)

        """ image augmentation """
        # Note: there is also patch based augmentation
        imgs, labs = utils_augmentation.augment_image(imgs,
                                                      labs,
                                                      channels_out,
                                                      b_mirror=b_mirror,
                                                      b_rotate=b_rotate,
                                                      b_scale=b_scale,
                                                      b_warp=b_warp,
                                                      b_permute_labels=b_permute_labels,
                                                      angle_max=angle_max,
                                                      scale_factor=scale_factor,
                                                      delta_max=delta_max)

        """ patch cropping """
        # perform random crop
        img_shape = tf.shape(imgs[0])
        max_crop_pos = img_shape - shape_input - 1
        crop_pos_offset = [int(x / 2) for x in shape_input]
        field_reduce = [int((a - b) / 2) for a, b in
                        zip(shape_input, shape_output)]  # Note: the output size has to correspond to your model output

        """ class 0 to channels_out-1 patch(es) """
        patch_images = list()
        patch_labels = list()
        patch_pos = list()

        for idx_class in range(channels_out):
            for idx_amount in range(patches_per_class[idx_class]):
                # fetch position
                positions_valid = tf.cast(tf.where(tf.equal(labs[0], idx_class)), dtype=tf.int32)
                rand_ind_temp = tf.squeeze(tf.random_uniform([1], minval=0, maxval=tf.shape(positions_valid)[0], dtype=tf.int32))
                start_pos_i = tf.minimum(max_crop_pos, tf.maximum([0, 0, 0], positions_valid[rand_ind_temp, :] - crop_pos_offset))
                start_pos_l = start_pos_i + field_reduce
                pos_temp = tf.cast((tf.divide(start_pos_i + crop_pos_offset, shape_padded_image) - 0.5) * 2, dtype=tf.float32)
                patch_pos.append(pos_temp + tf.random_normal([3], mean=0.0, stddev=sigma_pos, dtype=tf.float32))

                # crop input & output
                p_imgs = [None for _ in range(n_images)]
                for idx in range(n_images):
                    p_imgs[idx] = tf.slice(imgs[idx], start_pos_i, shape_input)

                    # tamper with patch
                    offsets_temp = tf.random_normal([1], mean=0.0, stddev=sigma_offset, dtype=tf.float32)
                    noise_temp = tf.random_normal(shape_input, mean=0.0, stddev=sigma_noise, dtype=tf.float32)
                    p_imgs[idx] = p_imgs[idx] + offsets_temp + noise_temp

                patch_images.append(tf.stack(p_imgs, axis=3))

                # Stack labels in channel last format
                p_labs = [None for _ in range(n_labels)]
                for idx in range(n_labels):
                    p_labs[idx] = tf.slice(labs[idx], start_pos_l, shape_output)

                patch_labels.append(tf.one_hot(p_labs[0], depth=channels_out, axis=3))
                # TODO: atm just labs[0] is effectively used

        if b_verbose:
            patch_images[0] = tf.Print(patch_images[0],
                                       [tf.shape(patch_images), tf.shape(patch_labels), tf.shape(patch_pos)],
                                       'produced patch content: ',
                                       summarize=10)

        content = (patch_images, patch_labels, patch_pos)

        return content

    with tf.name_scope('pipeline'):
        with tf.device('/cpu:*'):  # all dataset ops should be processed on the cpu
            size_input = len(filenames[0])

            # generate placeholders that receive paths of type str
            in_images = list()
            in_labels = list()
            for idx_img in range(len(filenames[0])):
                in_images.append(tf.data.Dataset.from_tensor_slices(filenames[0][idx_img]))  # automatically creates const variable! so be careful
            for idx_lab in range(len(filenames[1])):
                in_labels.append(tf.data.Dataset.from_tensor_slices(filenames[1][idx_lab]))
            dataset = tf.data.Dataset.zip((*in_images, *in_labels))  # zip lists of input images and labels

            # shuffle filenames - NOT tfrecords
            if b_shuffle:
                dataset = dataset.shuffle(buffer_size=size_input, reshuffle_each_iteration=True)

            if repeat > 0:
                dataset = dataset.repeat(count=repeat)


            # convert to tfrecord after shuffle
            dataset = dataset.flat_map(lambda *args: tf.data.Dataset.zip((tuple([tf.data.TFRecordDataset(arg) for arg in args]))))

            # map parse function to each zipped element
            dataset = dataset.map(map_func=_map_data, num_parallel_calls=num_parallel_calls)

            # unwrap patches into one big dataset containing pairs
            dataset = dataset.flat_map(lambda p_i, p_l, pos: tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(p_i),
                                                                                  tf.data.Dataset.from_tensor_slices(p_l),
                                                                                  tf.data.Dataset.from_tensor_slices(pos))))

            # shuffle enough patches, so that they appear randomly selected within a batch
            dataset = dataset.shuffle(buffer_size=size_batch * 3, seed=None,
                            reshuffle_each_iteration=True)

            # here one could use shuffle, repeat, prefetch, ...
            dataset_batched = dataset.batch(batch_size=size_batch,
                                            drop_remainder=True)  # > 1.8.0: use drop_remainder=True

            if size_buffer > 0:
                dataset_batched = dataset_batched.prefetch(buffer_size=size_buffer)

            iterator = dataset_batched.make_initializable_iterator()
            init_op_iterator = iterator.initializer

            # fetch next dataset element - corresponds to _map_data function
            images, labels, positions = iterator.get_next()

            images = tf.Print(images, [tf.shape(images), tf.shape(labels), positions], 'Passed pipeline content: ', summarize=25)

            # anything you pass here, will be accessible by the model
            spec_pipeline = {'images': images, 'labels': labels, 'positions': positions, 'init_op_iter': init_op_iterator}

            return spec_pipeline


def gen_pipeline_eval_patch(filenames,
                            shape_image,
                            shape_input,
                            shape_output,
                            channels_out,
                            size_batch,
                            size_buffer=1,
                            num_parallel_calls=1,
                            b_with_labels=False,
                            b_verbose=False):
    """ generates the tf data pipeline
        makes use of tfrecords
    """

    # generate values needed during mapping
    paddings_image = [int((a - b) / 2) for a, b in zip(shape_input, shape_output)]
    shape_padded_image = [a + 2 * b for a, b in zip(shape_image, paddings_image)]
    paddings_tiles = [int(((b - (a % b)) % b) / 2) for a, b in zip(shape_image, shape_output)]
    shape_padded_label = [a + 2 * b for a, b in zip(shape_image, paddings_tiles)]
    tiles = [int((a / b)) for a, b in zip(shape_padded_label, shape_output)]
    n_tiles = np.prod(tiles)



    def _map_data(*args):

        # order input
        n_images = len(filenames[0])
        if b_with_labels:
            n_labels = len(filenames[1])
        else:
            n_labels = 0

        imgs = list(args[0:n_images])
        labs = list(args[n_images:n_images+n_labels])  # allows for further labels

        # check that the correct data / shapes are provided - remember it's all a graph, so use special tf.Print magic
        if b_verbose:
            imgs[0] = tf.Print(imgs[0], [tf.shape(imgs[0])], 'fetched data shapes: ', summarize=5)

        # process imgs
        for idx in range(n_images):
            # read tf records
            imgs[idx] = convert_tf.parse_function(imgs[idx])

            # fetch shape of single image
            shape_temp = tf.shape(imgs[idx])

            # ensure correct dims if using whole images
            pos_begin = tf.cast(tf.divide(tf.subtract(shape_temp, shape_image), 2), dtype=tf.int32)
            imgs[idx] = tf.slice(imgs[idx], pos_begin, shape_image)

            # cast images
            imgs[idx] = tf.cast(imgs[idx], tf.float32)

            # image normalization:
            tmp_mean, tmp_var = tf.nn.moments(imgs[idx], axes=[0, 1, 2])
            imgs[idx] = tf.divide(imgs[idx] - tmp_mean, tf.maximum(tf.sqrt(tmp_var), tf.divide(1.0, tf.sqrt(tf.cast(tf.size(imgs[idx]), dtype=tf.float32)))))

        # stack everything in channel last format
        images = tf.stack(imgs, axis=3)

        if b_with_labels:
            for idx in range(len(labs)):
                labs[idx] = convert_tf.parse_function(labs[idx])

                shape_temp = tf.shape(labs[idx])
                pos_begin = tf.cast(tf.divide(tf.subtract(shape_temp, shape_image), 2), dtype=tf.int32)
                labs[idx] = tf.slice(labs[idx], pos_begin, shape_image)
                # b = tf.cast(b, tf.float32)

                # process labels
                labs[idx] = tf.clip_by_value(tf.cast(labs[idx], dtype=tf.int32), 0, channels_out - 1)

        else:
            labs = [tf.zeros_like(imgs[0], dtype=tf.int32)]

        labels = tf.one_hot(labs[0], depth=channels_out, axis=3)
        # TODO: atm just labs[0] is effectively used

        if b_verbose:
            images = tf.Print(images, [tf.shape(images), tf.shape(labels)], 'fetched images and labels: ', summarize=5)

        """ patch cropping """
        # use image to block routines of tf
        # TODO: the patch conversion requires atm a fixed predefined shape (i.e. as parameter)
        #       -> investigate dynamic options
        patches_image, patches_labels, positions = utils_patching.space_to_batch(images,
                                                                                 labels,
                                                                                 tiles,
                                                                                 n_tiles,
                                                                                 paddings_image=paddings_image,
                                                                                 paddings_tiles=paddings_tiles,
                                                                                 shape_padded_image=shape_padded_image,
                                                                                 shape_padded_label=shape_padded_label,
                                                                                 shape_input=shape_input,
                                                                                 shape_output=shape_output,
                                                                                 b_with_labels=b_with_labels,
                                                                                 b_verbose=b_verbose)

        # fill last batch with first few entries (again)
        # required to make dynamic conv work since batch_size has to be known at graph creation time
        remainder = patches_image.get_shape()[0] % size_batch
        say = int(patches_image.get_shape()[0]) / size_batch #check the value of whole batches
        print('asdddddddddddddddddddddddd',remainder,'haha',say)
        if b_verbose:
            print(f'patch remainder: {remainder}')

        if not remainder == 0:
            dummies = size_batch - remainder
            patches_image = tf.concat([patches_image, patches_image[:dummies, ...]], axis=0)
            patches_labels = tf.concat([patches_labels, patches_labels[:dummies, ...]], axis=0)
            positions = tf.concat([positions, positions[:dummies, ...]], axis=0)

        if b_verbose:
            patches_image = tf.Print(patches_image, [tf.shape(patches_image), tf.shape(patches_labels)], 'batched images and labels: ', summarize=5)

        # make tensors out of shape_padded_label and n_tiles so they can be split
        # shape_padded_tensor = tf.tile(tf.expand_dims(shape_padded_label, axis=0), [tf.shape(images)[0], 1])
        # n_tiles_tensor = tf.tile(tf.expand_dims([n_tiles], axis=0), [tf.shape(images)[0], 1])

        assert (remainder ==0)

        return patches_image, patches_labels, positions

    with tf.name_scope('pipeline'):
        with tf.device('/cpu:*'):  # all dataset ops should be processed on the cpu (faster - rly?)

            idx_sel = tf.placeholder(tf.int64, shape=[])  # selects which eval subject to feed to the data pipeline

            # generate placeholders that receive paths of type str
            in_image_files = tf.constant(filenames[0], tf.string)
            in_label_files = tf.constant(filenames[1], tf.string)

            in_images = list()
            in_labels = list()
            for idx_img in range(len(filenames[0])):
                in_images.append(tf.data.Dataset.from_tensors(in_image_files[idx_img][idx_sel]))  # automatically creates const variable! so be careful
            if b_with_labels:
                for idx_lab in range(len(filenames[1])):
                    in_labels.append(tf.data.Dataset.from_tensors(in_label_files[idx_lab][idx_sel]))

            dataset = tf.data.Dataset.zip((*in_images, *in_labels))  # zip lists of input images and labels

            # convert to tfrecord - done so it's more similar to training pipeline / alleviate further changes
            dataset = dataset.flat_map(lambda *args: tf.data.Dataset.zip((tuple([tf.data.TFRecordDataset(arg) for arg in args]))))

            # Note: already produces batches
            dataset = dataset.map(map_func=_map_data, num_parallel_calls=num_parallel_calls)

            # unwrap patches into one big dataset containing pairs
            if b_with_labels:
                dataset = dataset.flat_map(lambda images_, labels_, pos_:
                                           tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(images_),
                                                                tf.data.Dataset.from_tensor_slices(labels_),
                                                                tf.data.Dataset.from_tensor_slices(pos_))))
            else:
                dataset = dataset.flat_map(lambda images_, labels_, pos_:
                                           tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(images_),
                                                                tf.data.Dataset.from_tensors([0]).repeat(count=images_.get_shape()[0]),  # dummy value | tf.data.Dataset.from_tensors(labels_).repeat(count=images_.get_shape()[0])
                                                                tf.data.Dataset.from_tensor_slices(pos_))))

            # make batches out of the single input patches
            dataset_batched = dataset.batch(batch_size=size_batch,
                                            drop_remainder=True)  # all batches have to be processed

            if size_buffer > 0:
                dataset_batched = dataset_batched.prefetch(buffer_size=size_buffer)  # not sure what this does

            iterator = dataset_batched.make_initializable_iterator()
            init_op_iterator = iterator.initializer

            # fetch next dataset element - corresponds to _map_data function
            images, labels, positions = iterator.get_next()

            if b_verbose:
                images = tf.Print(images, [tf.shape(images), tf.shape(labels), positions], 'Passed pipeline content: ', summarize=25)

            # anything you pass here, will be accessible by the model
            spec_pipeline = {'images': images,
                             'labels': labels,
                             'init_op_iter': init_op_iterator,
                             'idx_selection': idx_sel,
                             'shape_padded_label': shape_padded_label,
                             'shape_image': shape_image,
                             'shape_output': shape_output,
                             'n_tiles': n_tiles,
                             'tiles': tiles,
                             'positions': positions}

            return spec_pipeline


def gen_pipeline_eval_image(filenames,
                            shape_image,
                            channels_out,
                            size_batch,
                            size_buffer=1,
                            num_parallel_calls=1,
                            b_with_labels=False,
                            b_verbose=True):
    """ generates the tf data pipeline
        makes use of tfrecords
    """

    def _map_data(*args):

        # order input
        n_images = len(filenames[0])
        n_labels = len(filenames[1])

        imgs = list(args[0:n_images])
        labs = list(args[n_images:n_images+n_labels])  # allows for further labels
        # Note: current processing just considers labs[0]

        # check that the correct data / shapes are provided - remember it's all a graph, so use special tf.Print magic
        if b_verbose:
            imgs[0] = tf.Print(imgs[0], [tf.shape(imgs[0])], 'fetched data shapes: ', summarize=5)

        for idx in range(n_images):
            # read tf records
            imgs[idx] = convert_tf.parse_function(imgs[idx])

            # fetch shape of single image
            shape_temp = tf.shape(imgs[idx])

            # ensure correct dims if using whole images
            pos_begin = tf.cast(tf.divide(tf.subtract(shape_temp, shape_image), 2), dtype=tf.int32)
            imgs[idx] = tf.slice(imgs[idx], pos_begin, shape_image)

            # cast images
            imgs[idx] = tf.cast(imgs[idx], tf.float32)

            # image normalization:
            tmp_mean, tmp_var = tf.nn.moments(imgs[idx], axes=[0, 1, 2])
            imgs[idx] = tf.divide(imgs[idx] - tmp_mean,
                                  tf.maximum(tf.sqrt(tmp_var), tf.divide(1.0, tf.sqrt(tf.cast(tf.size(imgs[idx]), dtype=tf.float32)))))

        # stack everything in channel last format
        images = tf.stack(imgs, axis=3)

        if b_with_labels:
            for idx in range(len(labs)):
                labs[idx] = convert_tf.parse_function(labs[idx])

                shape_temp = tf.shape(labs[idx])
                pos_begin = tf.cast(tf.divide(tf.subtract(shape_temp, shape_image), 2), dtype=tf.int32)
                labs[idx] = tf.slice(labs[idx], pos_begin, shape_image)
                # b = tf.cast(b, tf.float32)

                # process labels
                labs[idx] = tf.clip_by_value(tf.cast(labs[idx], dtype=tf.int32), 0, channels_out - 1)

        else:
            labs = [tf.zeros_like(imgs[0], dtype=tf.int32)]

        labels = tf.one_hot(labs[0], depth=channels_out, axis=3)
        # TODO: atm just labs[0] is effectively used

        if b_verbose:
            images = tf.Print(images, [tf.shape(images), tf.shape(labels)], 'fetched images and labels: ', summarize=5)



        return images, labels

    with tf.name_scope('pipeline'):
        with tf.device('/cpu:*'):  # all dataset ops should be processed on the cpu

            idx_sel = tf.placeholder(tf.int64, shape=[])  # selects which eval subject to feed to the data pipeline

            # generate placeholders that receive paths of type str
            in_image_files = tf.constant(filenames[0], tf.string)
            in_label_files = tf.constant(filenames[1], tf.string)

            in_images = list()
            in_labels = list()
            for idx_img in range(len(filenames[0])):
                in_images.append(tf.data.Dataset.from_tensors(in_image_files[idx_img][idx_sel]))  # automatically creates const variable! so be careful
            if b_with_labels:
                for idx_lab in range(len(filenames[1])):
                    in_labels.append(tf.data.Dataset.from_tensors(in_label_files[idx_lab][idx_sel]))

            dataset = tf.data.Dataset.zip((*in_images, *in_labels))  # zip lists of input images and labels

            # convert to tfrecord - done so it's more similar to training pipeline / alleviate further changes
            dataset = dataset.flat_map(lambda *args: tf.data.Dataset.zip((tuple([tf.data.TFRecordDataset(arg) for arg in args]))))

            # map parse function to each zipped element
            dataset = dataset.map(map_func=_map_data, num_parallel_calls=num_parallel_calls)

            # here one could use shuffle, repeat, prefetch, ...
            dataset_batched = dataset.batch(batch_size=size_batch, drop_remainder=True)

            if size_buffer > 0:
                dataset_batched = dataset_batched.prefetch(buffer_size=size_buffer)

            iterator = dataset_batched.make_initializable_iterator()
            init_op_iterator = iterator.initializer

            # fetch next dataset element - corresponds to _map_data function
            images, labels = iterator.get_next()

            # anything you pass here, will be accessible by the model
            spec_pipeline = {'images': images, 'labels': labels, 'init_op_iter': init_op_iterator, 'idx_selection': idx_sel}



            return spec_pipeline


def add_background(labels):
    background = tf.cast(tf.logical_not(tf.reduce_any(tf.cast(labels, tf.bool), axis=-1, keepdims=True)), tf.int32)

    return tf.concat([background, labels], axis=-1)
