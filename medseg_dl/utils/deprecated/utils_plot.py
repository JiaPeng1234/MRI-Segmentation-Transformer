import numpy as np
import matplotlib.pyplot as plt
import math
import statistics

# TODO: this script is not finished
#       it has just been cobbled together from old files
#       proper migration is pending


def calc_mean(scores=None):

    mean_scores = []
    for idx_inner_element in range(len(scores[0])):
        temp_score = 0
        for idx_list in range(len(scores)):
            temp_score += scores[idx_list][idx_inner_element]
        mean_scores.append( temp_score/len(scores) )

    return mean_scores


if __name__ == '__main__':

    var = 0.9
    var2 = 0.6
    mean = 0.85
    # dice
    subjects_dice = [[] for _ in range(11)]
    subjects_dice[0] = [0.9349550091728871, 0.8876646391335822, 0.8386781675335356, 0.8257490480065555]
    subjects_dice[1] = [0.9527993011958851, 0.9494874071682273, 0.9168707358460011, 0.9364019867501082]
    subjects_dice[2] = [0.9213865588909336, 0.9038869012470041, 0.8878574839113149, 0.9003728062462631]
    subjects_dice[3] = [0.9533589875543744, 0.9492648005021259, 0.9350267082689608, 0.9332716634557557]
    subjects_dice[4] = [0.9338449516929462, 0.9161155169833907, 0.9166371122636169, 0.9130147914101182]
    subjects_dice[5] = [0.9288151859431233, 0.9083383638679564, 0.9035048670250173, 0.9217033413697154]
    subjects_dice[6] = [0.8911468011276721, 0.8696815452884364, 0.8976773994436436, 0.9012220977819101]
    subjects_dice[7] = [0.8904451266308518, 0.8891478567752371, 0.830818654270937, 0.8705464611195023]
    subjects_dice[8] = [0.9507628992824451, 0.9305391795270925, 0.8906498228932535, 0.9070842815858715]
    subjects_dice[9] = [0.903242497412901, 0.9173333950853142, 0.904628599764115, 0.8545105879472846]
    subjects_dice[10] = [0.9511884165910248, 0.9498979708944132, 0.9191759853238453, 0.9097647587436407]
    for idx in range(11):
        subjects_dice[idx] = [sum(subjects_dice[idx])/4] + subjects_dice[idx]
    dice_coeffs = list(zip(*subjects_dice))

    # assd
    subjects_assd = [[] for _ in range(11)]
    subjects_assd[0] = [4.553744802932165, 6.040176164761251, 3.610250648318538, 3.3833427936154687]
    subjects_assd[1] = [1.8011355321718, 2.024267559987947, 1.2968979913709018, 1.0229905707967955]
    subjects_assd[2] = [6.871816641911275, 7.292095789988743, 3.2958282019209966, 2.721215999194751]
    subjects_assd[3] = [1.7734830976739924, 2.154026985768984, 1.2161739756781074, 1.660829418496454]
    subjects_assd[4] = [1.236551908937165, 2.2158730186493587, 1.6100896809811012, 2.176216005336289]
    subjects_assd[5] = [2.8528817774507402, 3.108403228199528, 1.7435276581389143, 1.5075880894255236]
    subjects_assd[6] = [3.953146374083555, 7.092077843243589, 2.8311920272566673, 3.450945952531415]
    subjects_assd[7] = [3.7586993264374176, 2.2930340579471133, 2.168589678942824, 1.5084759677059356]
    subjects_assd[8] = [1.0437909900740086, 2.403199922855203, 2.7987397899411457, 2.689387664973527]
    subjects_assd[9] = [7.301844810966932, 5.312020561546404, 1.2986420512944448, 2.0471283231695603]
    subjects_assd[10] = [1.5499429621524876, 2.0433505138529924, 1.3536438305989726, 2.2008571762452727]
    for idx in range(11):
        subjects_assd[idx] = [sum(subjects_assd[idx])/4] + subjects_assd[idx]
    assd_coeffs = list(zip(*subjects_assd))


    print('mean dsc 02:', subjects_dice[2])
    print('mean dsc 03:', subjects_dice[3])
    print('mean assd 02:', subjects_assd[2])
    print('mean dsc: ', sum(dice_coeffs[0])/11)
    print('mean assd: ', sum(assd_coeffs[0])/11)
    print('lowest mean:', sum(dice_coeffs[3])/11)
    print('highest assd:', sum(assd_coeffs[2])/11)
    print('highest median:', statistics.median(assd_coeffs[1]))

    aPos = np.array([1, 2, 3, 4, 5])
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 5.5))

    lBoxplot = []
    lBoxplot2 = []
    llBoxplot = [lBoxplot, lBoxplot2]
    lBoxplot.append(ax1.boxplot(dice_coeffs, positions=aPos, widths=0.25, showfliers=False, showmeans=True, patch_artist=True))

    lBoxplot2.append(ax2.boxplot(assd_coeffs, positions=aPos, widths=0.25, showfliers=False, showmeans=True, patch_artist=True))

    lBox_color = ['red', 'blue', 'darkblue', 'mediumseagreen', 'green']
    lBox_style = ['-', '--', '--', '-.', '-.']

    for iter_plot in range(len(llBoxplot)):
        for iter_box in range(len(llBoxplot[iter_plot])):
            for idx, box in enumerate(llBoxplot[iter_plot][iter_box]['boxes']):
                box.set_color(lBox_color[iter_box+idx])
                box.set_linestyle(lBox_style[iter_box+idx])
                box.set_linewidth(1)
                box.set_facecolor('whitesmoke')
            for idx, median in enumerate(llBoxplot[iter_plot][iter_box]['medians']):
                median.set_linewidth(1)
                median.set_color(lBox_color[iter_box+idx])
            for idx, mean in enumerate(llBoxplot[iter_plot][iter_box]['means']):
                mean.set_markersize(4)
                mean.set_marker('D')
                mean.set_markerfacecolor(lBox_color[iter_box+idx])
                mean.set_markeredgecolor('black')
            for idx, whisker in enumerate(llBoxplot[iter_plot][iter_box]['whiskers']):
                whisker.set_color(lBox_color[iter_box+math.floor(idx/2)])
                whisker.set_linestyle(lBox_style[iter_box+math.floor(idx/2)])
                whisker.set_linewidth(1)
            for idx, cap in enumerate(llBoxplot[iter_plot][iter_box]['caps']):
                cap.set_color(lBox_color[iter_box+math.floor(idx/2)])
                cap.set_linewidth(1)


    # fig1.legend([lBoxplot[0]['boxes'][0], lBoxplot[1]['boxes'][0], lBoxplot[2]['boxes'][0], lBoxplot[3]['boxes'][0], lBoxplot[4]['boxes'][0], lBoxplot[5]['boxes'][0]], ['unguided MV', 'unguided RF', 'voted MV', 'voted RF', 'guided MV', 'guided RF'], loc=9, bbox_to_anchor=(0.5, 1.02), fontsize=20, ncol= 6)
    ax1.tick_params(axis='both', labelsize=15)
    ax2.tick_params(axis='both', labelsize=15)

    ax1.set_xlim(0.4, 5.6)
    ax2.set_xlim(0.4, 5.6)
    ax1.set_ylim(0.8, 1)
    ax2.set_ylim(-0.1, 8)
    ax1.set_xlabel('bones', fontsize=20)
    ax1.set_ylabel('DSC', fontsize=20)
    ax2.set_xlabel('bones', fontsize=20)
    ax2.set_ylabel('ASSD [mm]', fontsize=20)

    ax1.grid(visible=True)
    ax2.grid(visible=True)

    ax1.set_yticks([0.80, 0.85, 0.90, 0.95, 1.00])
    ax1.set_xticks(aPos)
    ax2.set_xticks(aPos)
    ax1.set_xticklabels(('mean', 'femur r.', 'femur l.', 'pelvis r.', 'pelvis l.'))
    ax2.set_xticklabels(('mean', 'femur r.', 'femur l.', 'pelvis r.', 'pelvis l.'))

    plt.show()
