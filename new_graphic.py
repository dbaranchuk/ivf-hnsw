

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy
import re

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

dataset = "SIFT"
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
