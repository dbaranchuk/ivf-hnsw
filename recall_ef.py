cM16_M16 = '''
'''

cM16_M2 = '''
'''

cM2_M2 = '''1	0	1.9532 us	22.1327 dcs	3.8469 hps
2	0.0005	3.1669 us	30.2085 dcs	6.48821 hps
3	0.0015	5.8575 us	49.2623 dcs	7.76334 hps
4	0.0027	8.9787 us	65.0614 dcs	9.19949 hps
5	0.0052	12.6283 us	82.0762 dcs	10.5748 hps
6	0.0082	16.8302 us	99.8536 dcs	11.9697 hps
7	0.0112	24.5492 us	116.524 dcs	13.3138 hps
8	0.0137	25.2585 us	131.577 dcs	14.5773 hps
9	0.0158	29.431 us	146.258 dcs	15.8421 hps
10	0.0198	33.4852 us	159.052 dcs	16.9279 hps
11	0.0226	37.0964 us	170.535 dcs	17.9138 hps
12	0.0235	40.8137 us	181.197 dcs	18.8129 hps
13	0.0252	44.0193 us	191.374 dcs	19.6844 hps
14	0.027	51.0699 us	201.042 dcs	20.4989 hps
15	0.0279	50.2266 us	209.321 dcs	21.2009 hps
16	0.0297	53.9729 us	218.084 dcs	21.9643 hps
17	0.0316	56.7093 us	226.519 dcs	22.6992 hps
18	0.0327	59.9897 us	234.542 dcs	23.382 hps
19	0.0344	62.4632 us	241.912 dcs	24.0155 hps
20	0.0361	69.2978 us	249.535 dcs	24.6916 hps
21	0.0376	68.3735 us	257.181 dcs	25.3587 hps
22	0.0397	71.5546 us	264.313 dcs	25.9827 hps
23	0.0413	73.8921 us	271.496 dcs	26.6074 hps
24	0.0429	77.0735 us	277.956 dcs	27.1796 hps
25	0.0443	80.2457 us	284.057 dcs	27.719 hps
26	0.0453	83.3131 us	290.177 dcs	28.2476 hps
27	0.0473	88.542 us	296.335 dcs	28.7943 hps
28	0.0482	86.9214 us	302.578 dcs	29.3492 hps
29	0.0495	87.9962 us	308.073 dcs	29.8248 hps
30	0.0518	92.5501 us	314.357 dcs	30.3715 hps
40	0.0619	112.079 us	365.609 dcs	34.9423 hps
50	0.0683	132.124 us	409.211 dcs	38.8914 hps
60	0.0746	150.697 us	449.949 dcs	42.6021 hps
70	0.0798	162.033 us	487.3 dcs	46.0359 hps
80	0.085	174.739 us	520.819 dcs	49.1396 hps
90	0.0895	185.763 us	554.275 dcs	52.2512 hps
100	0.0933	198.861 us	587.552 dcs	55.3611 hps
140	0.1043	244.347 us	707.082 dcs	66.6817 hps
180	0.1157	293.372 us	817.545 dcs	77.3049 hps
220	0.124	344.164 us	921.806 dcs	87.4912 hps
260	0.1288	385.519 us	1024.04 dcs	97.6118 hps
300	0.1341	423.533 us	1123.65 dcs	107.537 hps
340	0.1383	465.276 us	1221.93 dcs	117.386 hps
380	0.1417	506.988 us	1317.67 dcs	127.071 hps
420	0.1457	548.158 us	1414.65 dcs	137.449 hps
460	0.1484	592.175 us	1509.31 dcs	147.924 hps
'''


import matplotlib.pyplot as plt
import numpy
import re


splitted_cM16_M16 = re.findall(r"[0-9.]+", cM16_M16_txt)
splitted_cM16_M2 = re.findall(r"[0-9.]+", cM16_M2_txt)
splitted_cM2_M2 = re.findall(r"[0-9.]+", cM2_M2_txt)

ef = range(1,30) + range(30, 100, 10) + range(100, 500, 40)


cM16_M16 = splitted_cM16_M16[1::5]
cM16_M2 = splitted_cM16_M2[1::5]
cM2_M2 = splitted_cM2_M2[1::5]

plt.plot(ef, cM16_M16, 'r--')
plt.plot(ef, cM16_M2)
plt.plot(ef, cM2_M2, 'g')

plt.axis([1, 460, 0, 1])
plt.xlabel('Ef', fontsize=14)
plt.ylabel('Recall@R', fontsize=14)
plt.text(ef[-1]+1, cM16_M16[-1], 'cM16_M16', fontsize=11, color=(1,0,0))
plt.text(ef[-1]+1, cM16_M2[-1], 'cM16_M2', fontsize=11, color=(0,0,1))
plt.text(ef[-1]+1, cM2_M2[-1], 'cM2_M2', fontsize=11, color=(0,1,0))

plt.title('efConstruction 240 Recall/Ef')
plt.savefig('recall_ef_smart.png')
#plt.s:ow()
