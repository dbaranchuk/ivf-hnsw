SIFT_O_IMI_txt = '''
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

SIFT_Subgroups_75_txt = '''
0.0021
0.0043
0.0092
0.0174
0.0341
0.0555
0.0877
0.1314
0.1876
0.2744
0.3821
0.5017
0.6198
0.7232
0.7994
0.8514
0.8819
0.899
0.9061
0.9079
0.9088
'''

O_IMI_txt = '''
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

Hybrid_txt = '''
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

Subgroups_50_txt = '''
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

Subgroups_75_txt = '''
0.0058
0.0106
0.0202
0.0374
0.0682
0.1103
0.1545
0.2066
0.2671
0.3506
0.4483
0.5581
0.6584
0.7465
0.8113
0.8554
0.8836
0.8992
0.9054
0.9084
0.909
'''

import matplotlib.pyplot as plt
import numpy
import re

k = range(21)

O_IMI = re.findall(r"[0-9.]+", SIFT_O_IMI_txt)
Subgroups_50 = re.findall(r"[0-9.]+", SIFT_Subgroups_50_txt)
Subgroups_75 = re.findall(r"[0-9.]+", SIFT_Subgroups_75_txt)
Hybrid = re.findall(r"[0-9.]+", SIFT_Hybrid_txt)

lineIMI, = plt.plot(k, O_IMI, 'r', label = 'O-IMI')
lineHybrid, = plt.plot(k, Hybrid, 'g', label = 'Hybrid')
lineSubgroups_50, = plt.plot(k, Subgroups_50, '--b', label = 'Subgroups Filter 50%')
lineSubgroups_75, = plt.plot(k, Subgroups_75, 'b', label = 'Subgroups Filter 75%')

plt.xticks(range(21))
plt.yticks(numpy.arange(0., 1.1, 0.1))

plt.axis([0, 20, 0, 1])
plt.xlabel('log2(R)', fontsize=12)
plt.ylabel('Recall@R', fontsize=12)

plt.title('SIFT1B')
plt.legend(fontsize=11, loc=2)
#plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
#           ncol=3, mode="expand", borderaxespad=0., prop={'size': 11})
plt.savefig('graphic.png')
#plt.s:ow()
