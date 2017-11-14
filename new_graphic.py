
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy
import seaborn as sns
sns.set(style='ticks', palette='Set2')
sns.despine()

SIFT_16_IMI_16384_recall = [0.329, 0.348, 0.353]
SIFT_16_IMI_16384_time = [2.56, 5.31, 8.46]

SIFT_16_IMI_4096_recall = [0.270, 0.307, 0.316]
SIFT_16_IMI_4096_time = [0.73, 1.27, 1.85]

SIFT_16_IVF_recall = [0.292, 0.331, 0.341]
SIFT_16_IVF_time = [0.53, 1.04, 1.47]

SIFT_16_IVF_Grouping_recall = [0.305, 0.349, 0.361]
SIFT_16_IVF_Grouping_time = [0.61, 1.14, 1.62]

SIFT_16_IVF_Pruning_recall = [0.330, 0.361, 0.369]
SIFT_16_IVF_Pruning_time = [0.69, 1.31, 2.03]

SIFT_8_IMI_16384_recall = [0.174, 0.177, 0.178]
SIFT_8_IMI_16384_time = [2.16, 3.95, 6.16]

SIFT_8_IMI_4096_recall = [0.145, 0.153, 0.155]
SIFT_8_IMI_4096_time = [0.56, 0.89, 1.25]

SIFT_8_IVF_recall = [0.146, 0.158, 0.161]
SIFT_8_IVF_time = [0.47, 0.83, 1.25]

SIFT_8_IVF_Grouping_recall = [0.167, 0.184, 0.188]
SIFT_8_IVF_Grouping_time = [0.52, 0.99, 1.42]

SIFT_8_IVF_Pruning_recall = [0.176, 0.187, 0.189]
SIFT_8_IVF_Pruning_time = [0.60, 1.15, 1.76]

dataset = "DEEP10"
l = 1
if dataset == "SIFT":
    # lineIMI_16384, = plt.plot(SIFT_16_IMI_16384_time[:l], SIFT_16_IMI_16384_recall[:l], 'r^', label = 'Inverted Multi-Index 16384$^2$')
    # lineIMI_4096, = plt.plot(SIFT_16_IMI_4096_time[:l], SIFT_16_IMI_4096_recall[:l], 'c^', label = 'Inverted Multi-Index 4096$^2$')
    # lineIVF, = plt.plot(SIFT_16_IVF_time[:l], SIFT_16_IVF_recall[:l], 'g^', label = 'Inverted Index 2$^{20}$')
    # lineGrouping, = plt.plot(SIFT_16_IVF_Grouping_time[:l], SIFT_16_IVF_Grouping_recall[:l], 'b^', label = 'Inverted Index\nGrouping 2$^{20}$')
    # linePruning, = plt.plot(SIFT_16_IVF_Pruning_time[:l], SIFT_16_IVF_Pruning_recall[:l], 'k^', label = 'Inverted Index\nGrouping + Pruning 2$^{20}$')
    #
    # lineIMI_16384, = plt.plot(SIFT_8_IMI_16384_time[:l], SIFT_8_IMI_16384_recall[:l], 'ro')
    # lineIMI_4096, = plt.plot(SIFT_8_IMI_4096_time[:l], SIFT_8_IMI_4096_recall[:l], 'co')
    # lineIVF, = plt.plot(SIFT_8_IVF_time[:l], SIFT_8_IVF_recall[:l], 'go')
    # lineGrouping, = plt.plot(SIFT_8_IVF_Grouping_time[:l], SIFT_8_IVF_Grouping_recall[:l], 'bo')
    # linePruning, = plt.plot(SIFT_8_IVF_Pruning_time[:l], SIFT_8_IVF_Pruning_recall[:l], 'ko')

    lineIMI_16384, = plt.plot(SIFT_8_IMI_16384_time[:l]+SIFT_16_IMI_16384_time[:l],
                              SIFT_8_IMI_16384_recall[:l]+SIFT_16_IMI_16384_recall[:l],
                              'r')#, label = 'Inverted Multi-Index 16384$^2$')

    lineIMI_4096, = plt.plot(SIFT_8_IMI_4096_time[:l]+SIFT_16_IMI_4096_time[:l],
                             SIFT_8_IMI_4096_recall[:l]+SIFT_16_IMI_4096_recall[:l],
                             'c')#, label = 'Inverted Multi-Index 4096$^2$')

    lineIVF, = plt.plot(SIFT_8_IVF_time[:l]+SIFT_16_IVF_time[:l],
                        SIFT_8_IVF_recall[:l]+SIFT_16_IVF_recall[:l],
                        'g')#, label = 'Inverted Index 2$^{20}$')

    lineGrouping, = plt.plot(SIFT_8_IVF_Grouping_time[:l]+SIFT_16_IVF_Grouping_time[:l],
                             SIFT_8_IVF_Grouping_recall[:l]+SIFT_16_IVF_Grouping_recall[:l],
                             'b')#, label = 'Inverted Index\nGrouping 2$^{20}$')

    linePruning, = plt.plot(SIFT_8_IVF_Pruning_time[:l]+SIFT_16_IVF_Pruning_time[:l],
                            SIFT_8_IVF_Pruning_recall[:l]+SIFT_16_IVF_Pruning_recall[:l],
                            'k')#, label = 'Inverted Index\nGrouping + Pruning 2$^{20}$')
    plt.xticks(numpy.arange(0., 7., 0.25))
    plt.yticks(numpy.arange(0., 0.51, 0.05))

    plt.axis([0, 2.6, 0.12, 0.351])
    plt.xlabel('Time', fontsize=11)
    plt.ylabel('Recall@1', fontsize=11)
    #plt.legend(fontsize=9, loc=1)

    pp = PdfPages('R@1_SIFT.pdf')
    pp.savefig()
    pp.close()



DEEP_16_IMI_16384_recall1 = [0.320, 0.359]
DEEP_16_IMI_16384_time = [1.89, 2.93]

DEEP_16_IMI_4096_recall1 = [0.245, 0.292]
DEEP_16_IMI_4096_time = [0.81, 1.23]

DEEP_16_IVF_recall1 = [0.349, 0.388]
DEEP_16_IVF_time = [0.51, 0.95]

DEEP_16_IVF_Grouping_recall1 = [0.369, 0.411]
DEEP_16_IVF_Grouping_time = [0.56, 1.04]

DEEP_16_IVF_Pruning_recall1 = [0.389, 0.421]
DEEP_16_IVF_Pruning_time = [0.63, 1.12]

DEEP_8_IMI_16384_recall1 = [0.196, 0.210]
DEEP_8_IMI_16384_time = [1.66, 2.39]

DEEP_8_IMI_4096_recall1 = [0.161, 0.179]
DEEP_8_IMI_4096_time = [0.56, 0.82]

DEEP_8_IVF_recall1 = [0.214, 0.228]
DEEP_8_IVF_time = [0.42, 0.78]

DEEP_8_IVF_Grouping_recall1 = [0.226, 0.241]
DEEP_8_IVF_Grouping_time = [0.46, 0.87]

DEEP_8_IVF_Pruning_recall1 = [0.234, 0.245]
DEEP_8_IVF_Pruning_time = [0.50, 1.01]


if dataset == "DEEP":
    lineIMI_16384, = plt.plot(DEEP_16_IMI_16384_time, DEEP_16_IMI_16384_recall1, 'r^', label = 'Inverted Multi-Index 16384$^2$')
    lineIMI_4096, = plt.plot(DEEP_16_IMI_4096_time, DEEP_16_IMI_4096_recall1, 'c^', label = 'Inverted Multi-Index 4096$^2$')
    lineIVF, = plt.plot(DEEP_16_IVF_time, DEEP_16_IVF_recall1, 'g^', label = 'Inverted Index 2$^{20}$')
    lineGrouping, = plt.plot(DEEP_16_IVF_Grouping_time, DEEP_16_IVF_Grouping_recall1, 'b^', label = 'Inverted Index\nGrouping 2$^{20}$')
    linePruning, = plt.plot(DEEP_16_IVF_Pruning_time, DEEP_16_IVF_Pruning_recall1, 'k^', label = 'Inverted Index\nGrouping + Pruning 2$^{20}$')

    lineIMI_16384, = plt.plot(DEEP_8_IMI_16384_time, DEEP_8_IMI_16384_recall1, 'ro')
    lineIMI_4096, = plt.plot(DEEP_8_IMI_4096_time, DEEP_8_IMI_4096_recall1, 'co')
    lineIVF, = plt.plot(DEEP_8_IVF_time, DEEP_8_IVF_recall1, 'go')
    lineGrouping, = plt.plot(DEEP_8_IVF_Grouping_time, DEEP_8_IVF_Grouping_recall1, 'bo')
    linePruning, = plt.plot(DEEP_8_IVF_Pruning_time, DEEP_8_IVF_Pruning_recall1, 'ko')
    plt.xticks(numpy.arange(0., 7., 0.2))
    plt.yticks(numpy.arange(0., 0.51, 0.05))

    plt.axis([0, 2, 0.12, 0.46])
    plt.xlabel('Time', fontsize=11)
    plt.ylabel('Recall@1', fontsize=11)
    plt.legend(fontsize=11, loc=1)

    pp = PdfPages('R@1_DEEP.pdf')
    pp.savefig()
    pp.close()

if dataset == "DEEP10":
    DEEP_16_IMI_16384_recall10 = [0.557, 0.671]
    DEEP_16_IMI_4096_recall10 = [0.431, 0.542]
    DEEP_16_IVF_recall10 = [0.612, 0.719]
    DEEP_16_IVF_Grouping_recall10 = [0.627, 0.736]
    DEEP_16_IVF_Pruning_recall10 = [0.679, 0.68]
    DEEP_8_IMI_16384_recall10 = [0.413, 0.457]
    DEEP_8_IMI_4096_recall10 = [0.320, 0.382]
    DEEP_8_IVF_recall10 = [0.447, 0.492]
    DEEP_8_IVF_Grouping_recall10 = [0.470, 0.519]
    DEEP_8_IVF_Pruning_recall10 = [0.496, 0.531]

    sns.set_style("ticks")
    lineIMI_16384_16, = plt.plot(DEEP_16_IMI_16384_time[:l], DEEP_16_IMI_16384_recall10[:l], 'r^', label = 'Inverted Multi-Index 16384$^2$')
    lineIMI_4096_16, = plt.plot(DEEP_16_IMI_4096_time[:l], DEEP_16_IMI_4096_recall10[:l], 'c^', label = 'Inverted Multi-Index 4096$^2$')
    lineIVF_16, = plt.plot(DEEP_16_IVF_time[:l], DEEP_16_IVF_recall10[:l], 'g^', label = 'Inverted Index 2$^{20}$')
    lineGrouping_16, = plt.plot(DEEP_16_IVF_Grouping_time[:l], DEEP_16_IVF_Grouping_recall10[:l], 'b^', label = 'Inverted Index Grouping 2$^{20}$')
    linePruning_16, = plt.plot(DEEP_16_IVF_Pruning_time[:l], DEEP_16_IVF_Pruning_recall10[:l], 'm^', label = 'Inverted Index Grouping + Pruning 2$^{20}$')

    lineIMI_16384_8, = plt.plot(DEEP_8_IMI_16384_time[:l], DEEP_8_IMI_16384_recall10[:l], 'ro', label = '')
    lineIMI_4096_8, = plt.plot(DEEP_8_IMI_4096_time[:l], DEEP_8_IMI_4096_recall10[:l], 'co',  label = '')
    lineIVF_8, = plt.plot(DEEP_8_IVF_time[:l], DEEP_8_IVF_recall10[:l], 'go', label = '')
    lineGrouping_8, = plt.plot(DEEP_8_IVF_Grouping_time[:l], DEEP_8_IVF_Grouping_recall10[:l], 'bo', label = '')
    linePruning_8, = plt.plot(DEEP_8_IVF_Pruning_time[:l], DEEP_8_IVF_Pruning_recall10[:l], 'mo', label = '')

    plt.xticks(numpy.arange(0., 7., 0.2))
    plt.yticks(numpy.arange(0., 1, 0.05))

    plt.axis([0, 2, 0.30, 0.71])
    plt.xlabel('Time', fontsize=11)
    plt.ylabel('Recall@10', fontsize=11)

    red_patch = mpatches.Patch(color='red', label='Inverted Multi-Index 16384$^2$')
    cyan_patch = mpatches.Patch(color='cyan', label='Inverted Multi-Index 4096$^2$')
    green_patch = mpatches.Patch(color='green', label='Inverted Index 2$^{20}$')
    blue_patch = mpatches.Patch(color='blue', label='Inverted Index Grouping 2$^{20}$')
    magenta_patch = mpatches.Patch(color='magenta', label='Inverted Index Grouping + Pruning 2$^{20}$')
    leg = plt.legend(frameon = False, fontsize=9, handles=[red_patch, cyan_patch, green_patch, blue_patch, magenta_patch], loc='best')

    PQ8, = plt.plot([], [], 'k^', label = '8 bytes')
    PQ16, = plt.plot([], [], 'ko',  label = '16 bytes')
    leg1 = plt.legend(frameon = False, fontsize=9, handles=[PQ8, PQ16], bbox_to_anchor=[0.5, 1], loc=1)

    # redLine = plt.plot([100], [100], 'r', label = 'Inverted Multi-Index 16384$^2$')
    # ceulLine, = plt.plot(DEEP_16_IMI_4096_time[:l], DEEP_16_IMI_4096_recall10[:l], 'c^', label = 'Inverted Multi-Index 4096$^2$')
    # greenLine, = plt.plot(DEEP_16_IVF_time[:l], DEEP_16_IVF_recall10[:l], 'g^', label = 'Inverted Index 2$^{20}$')
    # lineGrouping_16, = plt.plot(DEEP_16_IVF_Grouping_time[:l], DEEP_16_IVF_Grouping_recall10[:l], 'b^', label = 'Inverted Index\nGrouping 2$^{20}$')
    # linePruning_16, = plt.plot(DEEP_16_IVF_Pruning_time[:l], DEEP_16_IVF_Pruning_recall10[:l], 'k^', label = 'Inverted Index\nGrouping + Pruning 2$^{20}$')
    #leg1 = plt.legend((lineIMI_16384_8, lineIMI_4096_8, lineIVF_8, lineGrouping_8, linePruning_8), ['','','','',''], ncol=1, numpoints=1,
    #                   title='', handletextpad=-0.4,
    #                   bbox_to_anchor=[0.47, 1.], fontsize=9)
    # leg2 = plt.legend((lineIMI_4096_16, lineIMI_4096_8), ['', ''], ncol=1, numpoints=1,
    #                   title='Inverted Multi-Index 4096$^2$', handletextpad=-0.4,
    #                   bbox_to_anchor=[0.87, 1.], fontsize=12)
    # leg3 = plt.legend((lineIVF_16, lineIVF_8), ['', ''], ncol=1, numpoints=1,
    #                   title='Inverted Index 2$^{20}$', handletextpad=-0.4,
    #                   bbox_to_anchor=[0.99, 1.], fontsize=12)
    plt.gca().add_artist(leg)
    plt.gca().add_artist(leg1)
    # plt.gca().add_artist(leg3)

    pp = PdfPages('R@10_DEEP.pdf')
    pp.savefig(bbox_inches='tight')
    pp.close()