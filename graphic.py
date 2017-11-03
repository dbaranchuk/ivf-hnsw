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

O_IMI = re.findall(r"[0-9.]+", O_IMI_txt)
Subgroups_50 = re.findall(r"[0-9.]+", Subgroups_50_txt)
Subgroups_75 = re.findall(r"[0-9.]+", Subgroups_75_txt)
Hybrid = re.findall(r"[0-9.]+", Hybrid_txt)

lineIMI, = plt.plot(k, O_IMI, 'r', label = 'O-IMI')
lineHybrid, = plt.plot(k, Hybrid, 'g', label = 'Hybrid')
lineSubgroups_50, = plt.plot(k, Subgroups_50, '--b', label = 'Subgroups Filter 50%')
lineSubgroups_75, = plt.plot(k, Subgroups_75, 'b', label = 'Subgroups Filter 75%')

plt.xticks(range(21))
plt.yticks(numpy.arange(0., 1.1, 0.1))

plt.axis([0, 20, 0, 1])
plt.xlabel('log2(R)', fontsize=12)
plt.ylabel('Recall@R', fontsize=12)

plt.title('DEEP1B')
plt.legend(fontsize=11, loc=2)
#plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
#           ncol=3, mode="expand", borderaxespad=0., prop={'size': 11})
plt.savefig('graphic.png')
#plt.s:ow()
