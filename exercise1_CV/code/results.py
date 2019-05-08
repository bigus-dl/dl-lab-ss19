import numpy as np
import matplotlib.pyplot as plt
# import torch
import os
import pickle
import argparse

# from model.model import ResNetModel
# from model.data import get_data_loader
# from utils.plot_util import plot_keypoints
# from run_forward import normalize_keypoints


PATH = "results/fuckme.pth"
OPATH = "results/fuckyou.pth"

parser = argparse.ArgumentParser()
parser.add_argument("-m", action = "store_true", dest="pull_model", default = False)
parser.add_argument("-s", action = "store_true", dest="pull_samples", default = False)
args = parser.parse_args()

training_errors = []
validation_errors = []
mean_pixel_errors_val = []
mean_pixel_errors_train = []

if args.pull_samples:
    os.system("DEL /F/Q/S results\*.*")
    os.system("scp bahadorm@login1.informatik.uni-freiburg.de:~/Dokumente/dl-lab-ss19/exercise1_CV/code/results/*.png results/")
if args.pull_model :
    os.system("scp bahadorm@login1.informatik.uni-freiburg.de:~/Dokumente/dl-lab-ss19/exercise1_CV/code/model/*.pth results/")
os.system("scp bahadorm@login1.informatik.uni-freiburg.de:~/Dokumente/dl-lab-ss19/exercise1_CV/code/results/*.errors results/")

try:
    with open('results/training.errors', 'rb') as filehandle:
        training_errors = pickle.load(filehandle)
    with open('results/validation.errors', 'rb') as filehandle:
        validation_errors = pickle.load(filehandle)
    with open('results/training_pixel.errors', 'rb') as filehandle:
        mean_pixel_errors_train = pickle.load(filehandle)
    with open('results/validation_pixel.errors', 'rb') as filehandle:
        mean_pixel_errors_val = pickle.load(filehandle)
except FileNotFoundError:
    print("error file(s) not found")


epoch_eval = len(training_errors)/len(validation_errors)
training_errors = np.array(training_errors)
print("training error : {}\t last :{}".format(training_errors.shape,training_errors[-1]))
validation_errors = np.array(validation_errors).repeat(epoch_eval)
print("validation error : {}\t last: {}".format(validation_errors.shape,validation_errors[-1]))
mean_pixel_errors_val = np.array(mean_pixel_errors_val).repeat(epoch_eval)
print("MPJPE : {}\t last: {}".format(mean_pixel_errors_val.shape,mean_pixel_errors_val[-1]))
mean_pixel_errors_train = np.array(mean_pixel_errors_train)

plt.figure()
plt.plot(training_errors, label='training')
plt.plot(validation_errors, label='validation')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.title('HPE')
plt.legend()
plt.show()

plt.figure()
plt.plot(mean_pixel_errors_train, label='train.')
plt.plot(mean_pixel_errors_val, label='valid.')
plt.ylabel('mean pixel loss')
plt.xlabel('epochs')
plt.title('MPJPE')
plt.legend()
plt.show()