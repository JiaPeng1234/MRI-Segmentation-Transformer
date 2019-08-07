import numpy as np
import logging
from scipy.ndimage import morphology
from medseg_dl.utils import utils_misc
import matplotlib.pyplot as plt

# TODO: this script is not finished
#       it has just been cobbled together from old files
#       proper migration is pending


def get_bool_array(aInput, value_true):

    aInput_temp = np.copy(aInput)
    aInput_temp[aInput_temp != value_true] = 0
    aInput_temp = aInput_temp.astype(np.bool)

    return aInput_temp


def surfaceDistances(input1, input2, lSpacing=None, connectivity=1):
    """
    The distances between the surface voxel of binary objects in input1 and their
    nearest partner surface voxel of a binary object in input2.
    Author: Oskar Maier - MedPy - modified by MF
    """

    if lSpacing is not None:
        lSpacing = [1 for _ in range(input1.ndim)]

    input1 = np.atleast_1d(input1.astype(np.bool))
    input2 = np.atleast_1d(input2.astype(np.bool))

    # test for emptiness
    if 0 == np.count_nonzero(input1):
        input1 = np.invert(input1)
        logging.warning('The first supplied array does not contain any binary object.')
    if 0 == np.count_nonzero(input2):
        input2 = np.invert(input2)
        logging.warning('The second supplied array does not contain any binary object.')

    # binary structure
    footprint = morphology.generate_binary_structure(input1.ndim, connectivity)

    # extract only 1-pixel border line of objects
    input1_border = input1 ^ morphology.binary_erosion(input1, structure=footprint, iterations=1)
    input2_border = input2 ^ morphology.binary_erosion(input2, structure=footprint, iterations=1)

    # compute average surface distance
    # Note: scipys distance transform is calculated only inside the borders of the
    #       foreground objects, therefore the input has to be reversed
    dt = morphology.distance_transform_edt(~input2_border, sampling=lSpacing)
    sds = dt[input1_border]

    return sds


def calc_dice(label, pred, channels_out):

    lScore = [0 for _ in range(channels_out)]
    # TODO: convert to proper one-hot calculation
    for idx_class in range(channels_out):
        aInput_1_temp = get_bool_array(label, idx_class)
        aInput_2_temp = get_bool_array(pred, idx_class)

        nElements_input_1 = np.count_nonzero(aInput_1_temp)
        nElements_input_2 = np.count_nonzero(aInput_2_temp)
        nElements_and = np.count_nonzero(aInput_1_temp & aInput_2_temp)

        # calc dice
        if (nElements_input_1 + nElements_input_2) > 0:
            lScore[idx_class] = 2. * nElements_and / float(nElements_input_1 + nElements_input_2)
        else:
            lScore[idx_class] = 0.0

    return lScore


# calculates average symmetric surface distance
def calc_assd(label, pred, channels_out):

    lScore = [0 for _ in range(channels_out)]
    for idx_class in range(channels_out):
        aInput_1_temp = np.copy(label)
        aInput_2_temp = np.copy(pred)

        aInput_1_temp[aInput_1_temp != idx_class] = 0
        aInput_2_temp[aInput_2_temp != idx_class] = 0

        distances_surface_1 = surfaceDistances(aInput_1_temp, aInput_2_temp)
        distances_surface_2 = surfaceDistances(aInput_2_temp, aInput_1_temp)

        distance_surface_average_1 = distances_surface_1.mean()
        distance_surface_average_2 = distances_surface_2.mean()

        lScore[idx_class] = np.mean((distance_surface_average_1, distance_surface_average_2))

    return lScore


if __name__ == '__main__':

    path_image = '/media/sf_Promo/VBox_Folder/1811072/results_final/3/images__2018-11-07T01-14-41.npy'
    path_label = '/media/sf_Promo/VBox_Folder/1811072/results_final/3/labels__2018-11-07T01-14-41.npy'
    path_pred = '/media/sf_Promo/VBox_Folder/1811072/results_final/3/preds__2018-11-07T01-14-41.npy'
    channels_out = 5

    # load some arrays
    image = np.load(path_image)
    label = np.load(path_label)
    pred = np.load(path_pred)

    print(image.shape)
    print(label.shape)
    print(pred.shape)

    b_visualize = True
    if b_visualize:
        utils_misc.show_results(image[0, ...], label[0, ...], pred[0, ...])

    b_plot = True
    if b_plot:
        slice_y = 108
        slice_z = 22

        cmap = plt.cm.get_cmap('autumn', 4)
        cmap.set_under(color='k', alpha=0)
        im_label = cmap(np.argmax(label[0, ...], axis=-1)-1)
        im_pred = cmap(np.argmax(pred[0, ...], axis=-1)-1)

        # label xz
        fig1 = plt.figure(figsize=(12, 12))
        ax1 = plt.Axes(fig1, [0., 0., 1., 1.])
        ax1.set_axis_off()
        fig1.add_axes(ax1)
        ax1.imshow(np.rot90(image[0, :, slice_y, :, 0]), interpolation='none', cmap='gray', vmax=4)
        ax1.imshow(np.rot90(im_label[:, slice_y, :, :]), interpolation='none', alpha=0.5)
        plt.show()

        # pred xz
        fig2 = plt.figure(figsize=(12, 12))
        ax2 = plt.Axes(fig2, [0., 0., 1., 1.])
        ax2.set_axis_off()
        fig2.add_axes(ax2)
        ax2.imshow(np.rot90(image[0, :, slice_y, :, 0]), interpolation='none', cmap='gray', vmax=4)
        ax2.imshow(np.rot90(im_pred[:, slice_y, :, :]), interpolation='none', alpha=0.5)
        plt.show()

        # label xy
        fig1 = plt.figure(figsize=(12, 12))
        ax1 = plt.Axes(fig1, [0., 0., 1., 1.])
        ax1.set_axis_off()
        fig1.add_axes(ax1)
        ax1.imshow(np.rot90(image[0, :, :, slice_z, 0]), interpolation='none', cmap='gray', vmax=4)
        ax1.imshow(np.rot90(im_label[:, :, slice_z, :]), interpolation='none', alpha=0.5)
        plt.show()

        # pred xy
        fig2 = plt.figure(figsize=(12, 12))
        ax2 = plt.Axes(fig2, [0., 0., 1., 1.])
        ax2.set_axis_off()
        fig2.add_axes(ax2)
        ax2.imshow(np.rot90(image[0, :, :, slice_z, 0]), interpolation='none', cmap='gray', vmax=4)
        ax2.imshow(np.rot90(im_pred[:, :, slice_z, :]), interpolation='none', alpha=0.5)
        plt.show()

    b_calc = False
    if b_calc:
        dice_score = calc_dice(np.argmax(label[0, ...], axis=-1), np.argmax(pred[0, ...], axis=-1), channels_out)
        assd_score = calc_assd(np.argmax(label[0, ...], axis=-1), np.argmax(pred[0, ...], axis=-1), channels_out)

        print('dice scores: ', dice_score)
        print('assd scores: ', assd_score)
