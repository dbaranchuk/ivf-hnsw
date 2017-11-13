SIFT_O_IMI_16384_txt = '''
0.0029	
0.0054	
0.0099	
0.0175	
0.0326	
0.0531	
0.0874	
0.134	
0.202	
0.2908	
0.3979	
0.5213	
0.6388	
0.7532	
0.8464	
0.9154	
0.9612	
0.9831	
0.9948	
0.999	
0.9999	
'''

SIFT_O_IMI_4096_txt = '''
0.0004
0.001
0.0016
0.0033
0.0072
0.0137
0.0259
0.0466
0.0819
0.1338
0.2063
0.3041
0.4193
0.5398
0.6661
0.7804
0.8697
0.9345
0.9743
0.9888
0.9968
'''

SIFT_Hybrid_txt = '''
0.0004	
0.0004	
0.0005	
0.0014	
0.003	
0.0063	
0.0118	
0.0239	
0.0499	
0.0968	
0.1823	
0.301	
0.4321	
0.5706	
0.7045	
0.8142	
0.8903	
0.9447	
0.9743	
0.9906	
0.9969	
'''

SIFT_Subgroups_50_txt = '''
0.0021
0.0043
0.0092
0.0174
0.0341
0.0555
0.0877
0.1312
0.176
0.2273
0.3239
0.4386
0.5655
0.6894
0.7976
0.8733
0.9267
0.9565
0.9727
0.9795
0.9814
'''

SIFT_NEW_PRUNING_txt = '''
0.0001
0.0004
0.0015
0.0025
0.0048
0.0085
0.0148
0.0283
0.0531
0.1064
0.1899
0.3047
0.4337
0.5696
0.7049
0.8145
0.8913
0.9460
0.9754
0.9910
0.9973
'''

DEEP_O_IMI_16384_txt = '''
0.0023	
0.0047	
0.0099	
0.0169	
0.0271	
0.0421	
0.0616	
0.0872	
0.122	
0.1735	
0.2384	
0.3264	
0.429	
0.5503	
0.6711	
0.793	
0.8828	
0.9436	
0.9737	
0.9915	
0.9982
'''

DEEP_Hybrid_txt = '''
0.0003	
0.0012	
0.0018	
0.0035	
0.0059	
0.0109	
0.0199	
0.0368	
0.0717	
0.1415	
0.2611	
0.3886	
0.5109	
0.6402	
0.7506	
0.8442	
0.9105	
0.9524	
0.9786	
0.9915	
0.9967	
'''

DEEP_Subgroups_50_txt = '''
0.0058
0.0106
0.0202
0.0374
0.0681
0.1103
0.1544
0.2066
0.2532
0.3116
0.4011
0.5056
0.6226
0.7272
0.8197
0.8853
0.9299
0.9571
0.9709
0.9772
0.9796
'''

DEEP_O_IMI_4096_txt='''
0.0007
0.0014
0.0021
0.0047
0.0069
0.0124
0.0221
0.0353
0.0557
0.0876
0.1292
0.1842
0.2577
0.3575
0.4705
0.5893
0.7179
0.8377
0.922
0.9681
0.989
'''

DEEP_NEW_PRUNING_txt = '''
0.0003
0.0005
0.0009
0.002
0.0051
0.0105
0.0186
0.0367
0.0741
0.1485
0.2686
0.3889
0.513
0.6394
0.7501
0.8445
0.9106
0.9523
0.9787
0.9916
0.9968
'''

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy
import re

k = range(21)

dataset = "SIFT"
if dataset == "SIFT":
    O_IMI = re.findall(r"[0-9.]+", SIFT_O_IMI_16384_txt)
    Subgroups_50 = re.findall(r"[0-9.]+", SIFT_Subgroups_50_txt)
    O_IMI_4096 = re.findall(r"[0-9.]+", SIFT_O_IMI_4096_txt)
    Hybrid = re.findall(r"[0-9.]+", SIFT_Hybrid_txt)
    Pruning = re.findall(r"[0-9.]+", SIFT_NEW_PRUNING_txt)

    lineIMI, = plt.plot(k, O_IMI, '--r', label = 'Inverted Multi-Index 16384$^2$')
    lineIMI_4096, = plt.plot(k, O_IMI_4096, 'r', label = 'Inverted Multi-Index 4096$^2$')
    lineHybrid, = plt.plot(k, Hybrid, 'g', label = 'Inverted Index 2$^{20}$')
    #lineSubgroups_50, = plt.plot(k, Subgroups_50, 'b', label = 'Inverted Index\nGrouping + Pruning 2$^{20}$')
    linePruning, = plt.plot(k, Pruning, 'b', label = 'Inverted Index\nGrouping + Pruning 2$^{20}$')

    plt.xticks(range(0, 21, 2))
    plt.yticks(numpy.arange(0., 1.1, 0.1))

    plt.axis([0, 20, 0, 1])
    plt.xlabel('log$_2$R', fontsize=13)
    plt.ylabel('Recall@R', fontsize=13)
    plt.legend(fontsize=13, loc=2)

    pp = PdfPages('recallR_SIFT.pdf')
    pp.savefig()
    pp.close()
else:
    O_IMI = re.findall(r"[0-9.]+", DEEP_O_IMI_16384_txt)
    Subgroups_50 = re.findall(r"[0-9.]+", DEEP_Subgroups_50_txt)
    O_IMI_4096 = re.findall(r"[0-9.]+", DEEP_O_IMI_4096_txt)
    Hybrid = re.findall(r"[0-9.]+", DEEP_Hybrid_txt)
    Pruning = re.findall(r"[0-9.]+", DEEP_NEW_PRUNING_txt)

    lineIMI, = plt.plot(k, O_IMI, '--r', label = 'Inverted Multi-Index 16384$^2$')
    lineIMI_4096, = plt.plot(k, O_IMI_4096, 'r', label = 'Inverted Multi-Index 4096$^2$')
    lineHybrid, = plt.plot(k, Hybrid, 'g', label = 'Inverted Index 2$^{20}$')
    #lineSubgroups_50, = plt.plot(k, Subgroups_50, 'b', label = 'Inverted Index\nGrouping + Pruning 2$^{20}$')
    linePruning, = plt.plot(k, Pruning, 'b', label = 'Inverted Index\nGrouping + Pruning 2$^{20}$')

    plt.xticks(range(0, 21, 2))
    plt.yticks(numpy.arange(0., 1.1, 0.1))

    plt.axis([0, 20, 0, 1])
    plt.xlabel('log$_2$R', fontsize=13)
    plt.ylabel('Recall@R', fontsize=13)
    plt.legend(fontsize=13, loc=2)

    pp = PdfPages('recallR_DEEP.pdf')
    pp.savefig()
    pp.close()