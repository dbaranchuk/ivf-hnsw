import os
import matplotlib.pyplot as plt
import numpy

x = [1e+6, 2e+6, 5e+6, 1e+7, 2e+7, 5e+7, 1e+8]

one_1nn = [0.9987, 0.9968, 0.9938, 0.9911, 0.9856, 0.9752, 0.965]
rnd_1nn = [0.998, 0.9959, 0.9939, 0.9902, 0.9848, 0.979, 0.971]
one_10nn = [0.9955, 0.9929, 0.9884, 0.9829, 0.9749, 0.963, 0.9462]
rnd_10nn = [0.9953, 0.9919, 0.9871, 0.9818, 0.9748, 0.9653, 0.956]


plt.plot(x, one_10nn, 'r--')
plt.plot(x, rnd_10nn)
plt.axis([1e+6, 1e+8, 0.90, 1])

plt.xlabel('Dataset size')
plt.ylabel('recall@R')
plt.text(1e+8+1, one_10nn[-1], 'One Layer', fontsize=11, color=(1,0,0))
plt.text(1e+8+1, rnd_10nn[-1], 'HNSW', fontsize=11, color=(0,0,1))
plt.title('SIFT Experiment: 10NN')
plt.savefig('sift10NN.png')

#plt.show()
