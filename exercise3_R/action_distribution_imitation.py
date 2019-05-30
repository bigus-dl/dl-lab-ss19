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
    count['break'] = 0
    count['straight'] = 0

    for i in range(len(y)) :
        if np.allclose(y[i] ,    [-1 , 0.0, 0.0]) : count['left'] += 1
        elif np.allclose(y[i] ,  [+1 , 0.0, 0.0]) : count['right'] += 1
        elif np.allclose(y[i] ,  [0.0, 1.0, 0.0]) : count['accelerate'] += 1
        elif np.allclose(y[i] ,  [0.0, 0.0, 0.2]) : count['break'] += 1
        else : count['straight'] += 1

    plt.bar(range(len(count)), list(count.values()), align='center')
    plt.xticks(range(len(count)), list(count.keys()))
    plt.show()

    for x in count:
        print ("{} : {}".format(x , count[x]))

class_distribution()

'''
counts and weights
right : 4038 : 0,08076
break : 1118 : 0,02236
left : 8784 : 0,17568
straight : 22421 : 0,44842
accelerate : 13639 : 0,27278
sum : 50000
'''
