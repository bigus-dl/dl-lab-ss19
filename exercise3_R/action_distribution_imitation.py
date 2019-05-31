import torch
import pickle
import numpy as np
import os
import gzip
from PIL import Image
from imitation_learning.utils import *
import matplotlib.pyplot as plt

PKG_NAME = "data.pkl.gzip"

def class_distribution():
    print("decompressing data...")
    data_file = os.path.join("./imitation_learning/data", PKG_NAME)
  
    f = gzip.open(data_file,'rb')
    data = pickle.load(f)
    y = np.array(data["action"]).astype('float32')

    count = dict()
    count['left'] = 0
    count['right'] = 0
    count['accelerate'] = 0
    count['brake'] = 0
    count['straight'] = 0

    for i in range(len(y)) :
        if np.allclose(y[i] ,    [-1 , 0.0, 0.0]) : count['left'] += 1
        elif np.allclose(y[i] ,  [+1 , 0.0, 0.0]) : count['right'] += 1
        elif np.allclose(y[i] ,  [0.0, 1.0, 0.0]) : count['accelerate'] += 1
        elif np.allclose(y[i] ,  [0.0, 0.0, 0.2]) : count['brake'] += 1
        else : count['straight'] += 1

    plt.bar(range(len(count)), list(count.values()), align='center')
    plt.xticks(range(len(count)), list(count.keys()))
    plt.show()

    for x in count:
        print ("{} : count {}".format(x , count[x]))

'''
            left            right           accel.          brake           straight
count :     8784            4038            13639           1118            22421
normal:     0.17568         0.08076         0.27278         0.02236         0.44842
revers:     5.692167577     12.382367508    3.665957914     44.722719141    2.230052183
normal:     0.083708346     0.1820936398    0.053911145     0.6576870461    0.032794885
sftmax:     0.1724          0.1902          0.1674          0.3061          0.1639
sftmax:     0.1943          0.1978          0.1933          0.2221          0.1926
'''
class_distribution()
